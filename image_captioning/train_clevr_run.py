from train_clevr import *

if __name__ == '__main__':
    confs = []

    bs = 64
    lr = 1e-4

    for seed in [0]:
        # LSP
        for gcr_mode, safe_coef in [
            ('gc', None),
            ('gcr', 1e-1),
            ('gcr', 1e-2),
            ('gcr', 1e-3),
            ('gcr', 1e-4),
            ('gcr', 0),
        ]:
            confs.append(
                clevr_set(
                    type='set',
                    seed=seed,
                    batch_size=bs,
                    lr=lr,
                    gcr_mode=gcr_mode,
                    safe_coef=safe_coef,
                    track_switches=True,
                ))

        # Ordered set
        confs.append(clevr_set(
            type='seq',
            seed=seed,
            batch_size=bs,
            lr=lr,
        ))

        # Concat
        confs.append(clevr_text(
            seed=seed,
            batch_size=bs,
            lr=lr,
        ))

    multiprocess_map(Run(), confs)