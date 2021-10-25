from train_mimic import *

if __name__ == '__main__':
    confs = []

    n_max_length = 40
    n_max_sentences = 10
    bs = 64
    lr = 1e-4

    for seed in [0]:
        # LSP
        for (gcr_mode, safe_coef) in [
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
                    do_pad=False,
                    track_switches=True,
                    n_max_length=n_max_length,
                    n_max_sentences=n_max_sentences,
                    gcr_mode=gcr_mode,
                    safe_coef=safe_coef,
                    batch_size=bs,
                    lr=lr,
                ))

        # Ordered set
        confs.append(
            clevr_set(
                type='seq',
                seed=seed,
                do_pad=True,
                n_max_length=n_max_length,
                n_max_sentences=n_max_sentences,
                batch_size=bs,
                lr=lr,
                w_empty=0,
            ))

        # Concat
        confs.append(
            clevr_text(
                seed=seed,
                n_max_length=n_max_length,
                n_max_sentences=n_max_sentences,
                batch_size=bs,
                lr=lr,
            ))

    multiprocess_map(Run(), confs, debug=False)