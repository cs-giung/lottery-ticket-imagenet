import os
import sys
sys.path.append('./')

import math
import datetime
import itertools
import numpy as np
from tqdm import tqdm
from typing import Any
from tabulate import tabulate
from functools import partial
from collections import OrderedDict, namedtuple

import jax
import jaxlib
import flax
import optax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from flax import jax_utils, serialization
from flax.training import checkpoints, common_utils, train_state
from flax.training import dynamic_scale as dynamic_scale_lib
from tensorflow.io.gfile import GFile

from scripts import defaults
from giung2.data.tfds import input_pipeline
from giung2.data.torchvision import imagenet
from giung2.data.build import build_dataloader
from giung2.models.resnet import FlaxResNet
from giung2.metrics import evaluate_acc, evaluate_nll


def launch(config, print_fn):

    local_device_count = jax.local_device_count()
    shard_shape = (local_device_count, -1)

    # setup mixed precision training if specified
    platform = jax.local_devices()[0].platform
    if config.mixed_precision and platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()
        model_dtype = jnp.float16
    elif config.mixed_precision and platform == 'tpu':
        dynamic_scale = None
        model_dtype = jnp.bfloat16
    else:
        dynamic_scale = None
        model_dtype = jnp.float32

    # ----------------------------------------------------------------------- #
    # Dataset
    # ----------------------------------------------------------------------- #
    def prepare_tf_data(batch):
        batch['images'] = batch['images']._numpy()
        batch['labels'] = batch['labels']._numpy()
        batch['marker'] = np.ones_like(batch['labels'])
        def _prepare(x):
            if x.shape[0] < config.batch_size:
                x = np.concatenate([x, np.zeros([
                    config.batch_size - x.shape[0], *x.shape[1:]
                ], x.dtype)])
            return x.reshape(shard_shape + x.shape[1:])
        return jax.tree_util.tree_map(_prepare, batch)

    dataset_builder = tfds.builder(config.data_name)

    trn_split = 'train'
    trn_steps_per_epoch = math.ceil(
        dataset_builder.info.splits[trn_split].num_examples / config.batch_size)
    trn_iter = map(prepare_tf_data, input_pipeline.create_trn_split(
        dataset_builder, config.batch_size, split=trn_split))
    trn_iter = jax_utils.prefetch_to_device(trn_iter, config.prefetch_factor)

    val_split = 'validation'
    val_steps_per_epoch = math.ceil(
        dataset_builder.info.splits[val_split].num_examples / config.batch_size)
    val_iter = map(prepare_tf_data, input_pipeline.create_val_split(
        dataset_builder, config.batch_size, split=val_split))
    val_iter = jax_utils.prefetch_to_device(val_iter, config.prefetch_factor)

    NUM_CLASSES = 1000

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #
    model = FlaxResNet(
        image_size=224,
        depth=config.resnet_depth,
        widen_factor=config.resnet_width,
        dtype=model_dtype,
        pixel_mean=(0.48145466, 0.45782750, 0.40821073),
        pixel_std=(0.26862954, 0.26130258, 0.27577711))
        
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones((1, 224, 224, 3), model.dtype))
    variables = initialize_model(jax.random.PRNGKey(config.seed), model)

    # define forward function and specify shapes
    images = next(trn_iter)['images']
    output = jax.pmap(model.apply)({
        'params': jax_utils.replicate(variables['params']),
        'batch_stats': jax_utils.replicate(variables['batch_stats']),
        'image_stats': jax_utils.replicate(variables['image_stats'])}, images)
    FEATURE_DIM = output.shape[-1]

    log_str = f'images.shape: {images.shape}, output.shape: {output.shape}'
    print_fn(log_str)

    # setup trainable parameters
    if config.matching_ckpt is not None:

        matching_ckpt = checkpoints.restore_checkpoint(
            config.matching_ckpt, target=None)
        
        if config.previous_ckpt is None:
            previous_ckpt = matching_ckpt
        else:
            previous_ckpt = checkpoints.restore_checkpoint(
                config.previous_ckpt, target=None)

        if 'params_mask' not in previous_ckpt:
            previous_ckpt['params_mask'] = jax.tree_util.tree_map(
                jnp.ones_like, previous_ckpt['params'])

        # only the kernel parameters of convolutional layers will be pruned
        params_to_be_pruned = jax.tree_util.tree_map(
            jnp.ones_like if len(e.shape) == 4 else jnp.zeros_like,
            previous_ckpt['params'])

        # compute numbers
        params_to_be_pruned_flatten = \
            jax.flatten_util.ravel_pytree(params_to_be_pruned)[0]
        n_params_to_be_pruned = jnp.sum(params_to_be_pruned_flatten)
        n_params_not_to_be_pruned = \
            params_to_be_pruned_flatten.size - n_params_to_be_pruned
        n_params_to_be_pruned = jnp.sum(
            jax.flatten_util.ravel_pytree(previous_ckpt['params_mask'])[0]
        ) - n_params_not_to_be_pruned

        # compute per-element scores based on the magnitude
        params_score = jax.tree_util.tree_map(
            lambda e1, e2: (jnp.abs(e1) + 1) / e2 - 1,
            previous_ckpt['params'], params_to_be_pruned)

        # compute global threshold
        global_threshold = jax.lax.top_k(
            jax.flatten_util.ravel_pytree(params_score)[0],
            n_params_not_to_be_pruned
            + n_params_to_be_pruned * config.pruning_ratio)[0][-1]

        # get pruned parameters and masks
        params_mask = jax.tree_util.tree_map(
            lambda s: (s >= global_threshold).astype(jnp.float32),
            params_score)
        params = jax.tree_util.tree_map(
            lambda p, m: jnp.multiply(p, m),
            matching_ckpt['params'], params_mask)

    else:
        initial_ext_params = variables['params']
        initial_cls_params = jnp.zeros((FEATURE_DIM, NUM_CLASSES))
        params = {'ext': initial_ext_params, 'cls': initial_cls_params}
        params_mask = jax.tree_util.tree_map(jnp.ones_like, params)

    log_str = 'The number of trainable parameters: {:d} / {:d}'.format(
        int(jnp.sum(jax.flatten_util.ravel_pytree(params_mask)[0])),
        jax.flatten_util.ravel_pytree(params_mask)[0].size)
    print_fn(log_str)

    # ----------------------------------------------------------------------- #
    # Optimization
    # ----------------------------------------------------------------------- #
    # def step_trn(state, batch, config, scheduler, dynamic_scale):

    #     def _global_norm(updates):
    #         return jnp.sqrt(sum([jnp.sum(jnp.square(e))
    #                              for e in jax.tree_util.tree_leaves(updates)]))
        
    #     def _clip_by_global_norm(updates, global_norm):
    #         return jax.tree_util.tree_map(
    #             lambda e: jnp.where(
    #                 global_norm < config.optim_global_clipping, e,
    #                 (e / global_norm) * config.optim_global_clipping), updates)
        
    #     # define loss function
    #     def loss_fn(params):

    #         # get features
    #         output, new_model_state = model.apply({
    #             'params': jax.tree_util.tree_map(
    #                 lambda e1, e2: jnp.multiply(e1, e2),
    #                 params['ext'], state.params_mask['ext']),
    #             'batch_stats': state.batch_stats,
    #             'image_stats': state.image_stats}, batch['images'] / 255.0,
    #             mutable='batch_stats', use_running_average=False)

    #         # negative_log_likelihood
    #         smooth = config.optim_label_smoothing
    #         target = common_utils.onehot(batch['labels'], NUM_CLASSES)
    #         target = (1.0 - smooth) * target + \
    #             smooth * jnp.ones_like(target) / NUM_CLASSES
    #         source = jax.nn.log_softmax(output @ params['cls'], axis=-1)
    #         negative_log_likelihood = -jnp.sum(target * source, axis=-1)
    #         negative_log_likelihood = jnp.mean(negative_log_likelihood)

    #         # loss
    #         loss = negative_log_likelihood

    #         # log metrics
    #         metrics = OrderedDict({
    #             'loss': loss,
    #             'negative_log_likelihood': negative_log_likelihood})
    #         return loss, (metrics, new_model_state)

    #     # compute losses and gradients
    #     if dynamic_scale:
    #         dynamic_scale, is_fin, aux, grads = dynamic_scale.value_and_grad(
    #             loss_fn, has_aux=True, axis_name='batch')(state.params)
    #     else:
    #         aux, grads = jax.value_and_grad(
    #             loss_fn, has_aux=True)(state.params)
    #         grads = jax.lax.pmean(grads, axis_name='batch')

    #     # weight decay regularization in PyTorch-style
    #     grads = jax.tree_util.tree_map(
    #         lambda g, p: g + config.optim_weight_decay * p,
    #         grads, state.params)
        
    #     # mask gradients
    #     grads = jax.tree_util.tree_map(
    #         lambda e1, e2: jnp.multiply(e1, e2),
    #         grads, state.params_mask)

    #     # compute norms of weights and gradients
    #     w_norm = _global_norm(state.params)
    #     g_norm = _global_norm(grads)
    #     if config.optim_global_clipping:
    #         grads = _clip_by_global_norm(grads, g_norm)

    #     # get auxiliaries
    #     metrics = jax.lax.pmean(aux[1][0], axis_name='batch')
    #     metrics['w_norm'] = w_norm
    #     metrics['g_norm'] = g_norm
    #     metrics['lr'] = scheduler(state.step)

    #     # update train state
    #     new_state = state.apply_gradients(
    #         grads=grads, batch_stats=aux[1][1]['batch_stats'])
    #     if dynamic_scale:
    #         new_state = new_state.replace(
    #             opt_state=jax.tree_util.tree_map(
    #                 partial(jnp.where, is_fin),
    #                 new_state.opt_state, state.opt_state),
    #             params=jax.tree_util.tree_map(
    #                 partial(jnp.where, is_fin),
    #                 new_state.params, state.params))
    #         metrics['dyn_scale'] = dynamic_scale.scale
        
    #     # mask parameters
    #     new_state = new_state.replace(
    #         params=jax.tree_util.tree_map(
    #             lambda e1, e2: jnp.multiply(e1, e2),
    #             new_state.params, new_state.params_mask))
    #     return new_state, metrics
    
    # # define optimizer with scheduler
    # if config.constant_lr:
    #     scheduler = optax.constant_schedule(config.optim_lr)
    # else:
    #     scheduler = optax.join_schedules(
    #         schedules=[
    #             optax.linear_schedule(
    #                 init_value       = 0.0,
    #                 end_value        = config.optim_lr,
    #                 transition_steps = math.floor(0.1 * config.optim_ni)),
    #             optax.cosine_decay_schedule(
    #                 init_value       = config.optim_lr,
    #                 decay_steps      = math.floor(0.9 * config.optim_ni))
    #         ], boundaries=[
    #             math.floor(0.1 * config.optim_ni),
    #         ])
    # optimizer = optax.sgd(
    #     scheduler, momentum=config.optim_momentum,
    #     accumulator_dtype=model_dtype)

    # # build and replicate train state
    # class TrainState(train_state.TrainState):
    #     params_mask: Any = None
    #     batch_stats: Any = None
    #     image_stats: Any = None

    # state = TrainState.create(
    #     apply_fn=model.apply, params=params, tx=optimizer,
    #     params_mask=params_mask,
    #     batch_stats=variables['batch_stats'],
    #     image_stats=variables['image_stats'])
    # state = jax_utils.replicate(state)

    def apply_fn(images, state):
        return model.apply({
            'params': state.params['ext'],
            'batch_stats': state.batch_stats,
            'image_stats': state.image_stats,
        }, images, use_running_average=True) @ state.params['cls']
    p_apply_fn = jax.pmap(apply_fn)

    # # run optimization
    # best_acc = 0.0
    # p_step_trn = jax.pmap(partial(
    #     step_trn, config=config, scheduler=scheduler), axis_name='batch')
    # sync_batch_stats = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    
    # if dynamic_scale:
    #     dynamic_scale = jax_utils.replicate(dynamic_scale)

    # trn_metric = []
    # for iter_idx in itertools.count(start=1):
        
    #     # rendezvous
    #     jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    #     # terminate training
    #     if iter_idx == config.optim_ni + 1:
    #         break

    #     # ------------------------------------------------------------------- #
    #     # Train
    #     # ------------------------------------------------------------------- #
    #     log_str = '[Iter {:7d}/{:7d}] '.format(iter_idx, config.optim_ni)

    #     batch = next(trn_iter)
    #     state, metrics = p_step_trn(state, batch, dynamic_scale=dynamic_scale)
    #     trn_metric.append(metrics)

    #     if iter_idx % 1000 == 0:
    #         trn_summarized, val_summarized, tst_summarized = {}, {}, {}
            
    #         trn_metric = common_utils.get_metrics(trn_metric)
    #         trn_summarized = {f'trn/{k}': v for k, v in jax.tree_util.tree_map(
    #             lambda e: e.mean(), trn_metric).items()}
    #         trn_metric = []

    #         log_str += ', '.join(
    #             f'{k} {v:.3e}' for k, v in trn_summarized.items())

    #         # synchronize batch_stats across replicas
    #         state = state.replace(
    #             batch_stats=sync_batch_stats(state.batch_stats))

    #         # --------------------------------------------------------------- #
    #         # Valid
    #         # --------------------------------------------------------------- #
    #         acc, nll, cnt = 0.0, 0.0, 0
    #         for batch_idx, batch in enumerate(val_iter, start=1):
    #             logits = p_apply_fn(batch['images'] / 255.0, state)
    #             logits = logits.reshape(-1, NUM_CLASSES)
    #             labels = batch['labels'].reshape(-1)
    #             marker = batch['marker'].reshape(-1)
    #             pre = jax.nn.log_softmax(logits, axis=-1)
    #             acc += jnp.sum(jnp.where(marker, evaluate_acc(
    #                 pre, labels, log_input=True, reduction='none'
    #             ), marker))
    #             nll += jnp.sum(jnp.where(marker, evaluate_nll(
    #                 pre, labels, log_input=True, reduction='none'
    #             ), marker))
    #             cnt += jnp.sum(marker)
    #             if batch_idx == val_steps_per_epoch:
    #                 break
    #         val_summarized['val/acc'] = acc / cnt
    #         val_summarized['val/nll'] = nll / cnt
    #         val_summarized['val/best_acc'] = max(
    #             val_summarized['val/acc'], best_acc)

    #         log_str += ', '
    #         log_str += ', '.join(
    #             f'{k} {v:.3e}' for k, v in val_summarized.items())

    #         # --------------------------------------------------------------- #
    #         # Save
    #         # --------------------------------------------------------------- #
    #         if config.periodic_ckpt:

    #             curr_ckpt = {
    #                 'params': state.params,
    #                 'params_mask': state.params_mask,
    #                 'batch_stats': state.batch_stats,
    #                 'image_stats': state.image_stats}
    #             curr_ckpt = jax.device_get(
    #                 jax.tree_util.tree_map(lambda x: x[0], curr_ckpt))
                
    #             if config.save:
    #                 curr_path = os.path.join(
    #                     config.save, f'iter_{iter_idx:06d}.ckpt')
    #                 with GFile(curr_path, 'wb') as fp:
    #                     fp.write(serialization.to_bytes(curr_ckpt))

    #         if best_acc < val_summarized['val/acc']:

    #             log_str += ' (best_acc: {:.3e} -> {:.3e})'.format(
    #                 best_acc, val_summarized['val/acc'])
    #             best_acc = val_summarized['val/acc']

    #             best_ckpt = {
    #                 'params': state.params,
    #                 'params_mask': state.params_mask,
    #                 'batch_stats': state.batch_stats,
    #                 'image_stats': state.image_stats}
    #             best_ckpt = jax.device_get(
    #                 jax.tree_util.tree_map(lambda x: x[0], best_ckpt))

    #             if config.save:
    #                 best_path = os.path.join(config.save, 'best_acc.ckpt')
    #                 with GFile(best_path, 'wb') as fp:
    #                     fp.write(serialization.to_bytes(best_ckpt))
                
    #         # logging current iteration
    #         print_fn(log_str)

    #         # terminate training if loss is nan
    #         if jnp.isnan(trn_summarized['trn/loss']):
    #             break

    # --------------------------------------------------------------- #
    # ImageNetV2
    # --------------------------------------------------------------- #
    from flax.training import checkpoints
    best_ckpt = checkpoints.restore_checkpoint(
        f'./save/imagenet2012/IMP-LTR/{config.seed:03d}/best_acc.ckpt', target=None)

    best_ckpt = jax_utils.replicate(best_ckpt)
    best_params = namedtuple(
        'best_params', ['params', 'batch_stats', 'image_stats']
    )(best_ckpt['params'], best_ckpt['batch_stats'], best_ckpt['image_stats'])

    tst_summarized = {}
    for dataset_name in ['ImageNet', 'ImageNetV2',
                         'ImageNetR', 'ImageNetA', 'ImageNetSketch']:

        dataset = getattr(
            imagenet, dataset_name)(load_trn=False, load_val=False)

        tst_images = np.load(os.path.join(
            config.data_root, f'{dataset_name}_x224/test_images.npy'))
        tst_labels = np.load(os.path.join(
            config.data_root, f'{dataset_name}_x224/test_labels.npy'))
        dataloader = build_dataloader(
            tst_images, tst_labels, config.batch_size)
        tst_steps_per_epoch = math.ceil(
            tst_images.shape[0] / config.batch_size)

        acc, nll, cnt = 0.0, 0.0, 0
        for batch in tqdm(dataloader, total=tst_steps_per_epoch,
                          leave=False, ncols=0, desc=dataset_name):
            
            labels = jnp.array(batch['labels'])[:batch['marker'].sum()]
            images = jnp.array(batch['images'] / 255.0)
            if images.shape[0] != config.batch_size:
                images = jnp.zeros(
                    (config.batch_size,) + images.shape[1:]
                ).at[:images.shape[0]].set(images)
            images = images.reshape(shard_shape + images.shape[1:])

            logits = p_apply_fn(images, best_params)
            logits = logits.reshape(-1, 1000)[:labels.shape[0]]
            project_logits = getattr(dataset, 'project_logits', None)
            if project_logits is not None:
                logits = project_logits(logits)
            
            pre = jax.nn.softmax(logits, axis=-1)
            acc += evaluate_acc(pre, labels, log_input=False, reduction='sum')
            nll += evaluate_nll(pre, labels, log_input=False, reduction='sum')
            cnt += labels.shape[0]

        tst_summarized[f'{dataset_name}/acc'] = acc / cnt
        tst_summarized[f'{dataset_name}/nll'] = nll / cnt

    # logging current iteration
    log_str = ', '.join(f'{k} {v:.3e}' for k, v in tst_summarized.items())
    print_fn(log_str)


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument(
        '--optim_ni', default=64000, type=int,
        help='the number of training iterations (default: 64000)')
    parser.add_argument(
        '--optim_lr', default=0.8, type=float,
        help='base learning rate (default: 0.8)')
    parser.add_argument(
        '--optim_momentum', default=0.9, type=float,
        help='momentum coefficient (default: 0.9)')
    parser.add_argument(
        '--optim_weight_decay', default=0.0001, type=float,
        help='weight decay coefficient (default: 0.0001)')

    parser.add_argument(
        '--optim_label_smoothing', default=0.0, type=float,
        help='label smoothing regularization (default: 0.0)')
    parser.add_argument(
        '--optim_global_clipping', default=None, type=float,
        help='global norm for the gradient clipping (default: None)')

    parser.add_argument(
        '--save', default=None, type=str,
        help='save the *.log and *.ckpt files if specified (default: False)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='random seed for training (default: None)')

    parser.add_argument(
        '--matching_ckpt', default=None, type=str,
        help='path to the matching checkpoint (default: None)')
    parser.add_argument(
        '--previous_ckpt', default=None, type=str,
        help='path to the previous checkpoint (default: None)')
    parser.add_argument(
        '--pruning_ratio', default=1.0, type=float,
        help='how many prunable parameters will be preserved (default: 1.0)')

    parser.add_argument(
        '--constant_lr', default=False, type=defaults.str2bool,
        help='use constant learning rate (default: False)')
    parser.add_argument(
        '--periodic_ckpt', default=False, type=defaults.str2bool,
        help='save checkpoint periodically (default: False)')

    parser.add_argument(
        '--mixed_precision', default=False, type=defaults.str2bool,
        help='run mixed precision training if specified (default: False)')

    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big'))

    if args.save is not None:
        if os.path.exists(args.save):
            raise AssertionError(f'already existing args.save = {args.save}')
        os.makedirs(args.save, exist_ok=True)

    def print_fn(s):
        s = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ') + s
        if args.save is not None:
            with open(os.path.join(args.save, f'{TIME_STAMP}.log'), 'a') as fp:
                fp.write(s + '\n')
        print(s, flush=True)

    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__
            + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__
            + ' @' + os.path.dirname(jaxlib.__file__)),
        ('Flax', flax.__version__
            + ' @' + os.path.dirname(flax.__file__)),
        ('Optax', optax.__version__
            + ' @' + os.path.dirname(optax.__file__)),
    ]) + '\n'
    log_str = f'Environments:\n{log_str}'
    print_fn(log_str)

    log_str = ''
    max_k_len = max(map(len, vars(args).keys()))
    for k, v in vars(args).items():
        log_str += f'- args.{k.ljust(max_k_len)} : {v}\n'
    log_str = f'Command line arguments:\n{log_str}'
    print_fn(log_str)

    if jax.local_device_count() > 1:
        log_str = (
            'Multiple local devices are detected:\n'
            f'{jax.local_devices()}\n')
        print_fn(log_str)

    launch(args, print_fn)


if __name__ == '__main__':
    main()
