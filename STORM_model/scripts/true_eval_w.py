"""TRUE-metric eval with env_context override (see true_eval_forkwall.py).

    python -m scripts.true_eval_w --results-dir results/<id> --env-context 64 \\
        [--step N] [--episodes 600] [--sampled]
"""
from argparse import ArgumentParser

from cl.config import setup_environment
setup_environment()

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from cl.agents import load_agent
from cl.environments import make_environment
from cl.trainer.checkpoint import load_checkpoint
from cl.trainer.utils import RNGManager


def main():
    p = ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--results-dir',
                   help='a training results dir (results/<id>/checkpoints/...)')
    g.add_argument('--bundle',
                   help='a released bundle dir holding run_config.yaml + '
                        'checkpoint_step_<N>/ directly (e.g. final_models/storm)')
    p.add_argument('--env-context', type=int, default=64)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--episodes', type=int, default=600)
    p.add_argument('--num-envs', type=int, default=24)
    p.add_argument('--maps-path', default='data/bridge_tunnel/forkwall6k/test.pkl')
    p.add_argument('--sampled', action='store_true')
    p.add_argument('--seed', type=int, default=999)
    args = p.parse_args()

    if args.bundle:
        run_config_path = f'{args.bundle}/run_config.yaml'
    else:
        run_config_path = f'{args.results_dir}/checkpoints/run_config.yaml'
    run_config = OmegaConf.load(run_config_path)
    config = OmegaConf.merge(run_config, OmegaConf.create({
        'seed': args.seed,
        'agent': {'model': {'env_context': args.env_context}},
        'env': {'num_parallel_envs': args.num_envs,
                'num_parallel_envs_eval': args.num_envs,
                'maps_path': args.maps_path}}))
    agent = load_agent(config)
    state = agent.init(RNGManager(seed=args.seed).get_key())
    if args.bundle:
        # bundle layout: checkpoint dirs named checkpoint*_step_<N> at top level
        from pathlib import Path
        cands = sorted(Path(args.bundle).glob('checkpoint*step_*'))
        if args.step is not None:
            cands = [c for c in cands if c.name.endswith(f'{args.step:08d}')]
        if not cands:
            raise FileNotFoundError(f'no checkpoint dirs in {args.bundle}')
        import orbax.checkpoint as ocp
        ckpt = ocp.StandardCheckpointer().restore(cands[-1].resolve())
        meta = {'step': cands[-1].name.rsplit('_', 1)[-1]}
    else:
        ckpt, _, meta = load_checkpoint(
            checkpoint_dir=f'{args.results_dir}/checkpoints/BridgeTunnel/forkwall',
            step=args.step)
    state = agent.state_from_checkpoint(ckpt, state.runtime)
    print(f'ckpt step {meta.get("step")}  W={agent.env_context}  '
          f'mode={"sampled" if args.sampled else "greedy"}')

    env_cfg = OmegaConf.create({'seed': args.seed,
                                'env': OmegaConf.to_container(config.env, resolve=True)})
    env = make_environment('BridgeTunnel/forkwall', env_cfg)
    env_state = env.reset(None)
    prev_actions = jnp.zeros((env.num_envs, agent.action_space))
    rng = jax.random.PRNGKey(args.seed)

    stats = {c: {'correct': 0, 'wrong': 0, 'timeout': 0, 'n': 0}
             for c in ('balanced', 'lakes', 'rocky')}
    while sum(v['n'] for v in stats.values()) < args.episodes:
        obs = env_state.env_state.observation
        done = env_state.env_state.is_done()
        is_first = (env_state.env_state.t == 0) & (~done)
        rng, arng = jax.random.split(rng)
        acts, state = agent.select_action(state, obs, arng, is_first=is_first,
                                          prev_action=prev_actions,
                                          training=args.sampled)
        prev_actions = jax.nn.one_hot(acts, agent.action_space)
        cats = [env._envs[i]._record.category for i in range(env.num_envs)]
        env_state = env.step(env_state, acts)
        for i in np.where(np.asarray(env_state.env_state.is_done()))[0]:
            e = env._envs[i]
            s = stats[cats[i]]
            s['n'] += 1
            if e._pos in (e._correct_cells or set()):
                s['correct'] += 1
            elif e._step_count < e.max_steps:
                s['wrong'] += 1
            else:
                s['timeout'] += 1

    for c, v in stats.items():
        print(f"{c:9s}: correct {v['correct']}/{v['n']} = {v['correct']/max(1,v['n']):.3f}"
              f"   wrong {v['wrong']}   timeout {v['timeout']}")
    tot = {k: sum(v[k] for v in stats.values()) for k in ('correct', 'wrong', 'timeout', 'n')}
    print(f"TRUE success : {tot['correct']}/{tot['n']} = {tot['correct']/tot['n']:.4f}"
          f"   wrong {tot['wrong']/tot['n']:.3%}   timeout {tot['timeout']/tot['n']:.3%}")


if __name__ == '__main__':
    main()
