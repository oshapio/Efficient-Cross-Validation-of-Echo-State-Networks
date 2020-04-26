def get_n_walk_step_walking_splits(train_leng, train_slider_leng, folds_count=5, validation_leng=None):
    results = []

    # take biggest step s.t. we don't overstep the boundary
    step_size = (train_leng - train_slider_leng - validation_leng) // (folds_count - 1)

    accum_step = 0

    rem = (train_leng - train_slider_leng - validation_leng) % (folds_count - 1)

    for i in range(folds_count):
        results.append(
            {
                "exclude_train_range_start": (0, accum_step),
                "exclude_train_range_end": (accum_step + train_slider_leng, train_leng),

                "exclude_range": (accum_step + train_slider_leng, accum_step + train_slider_leng + validation_leng),
                "exclude_range_end": (0, accum_step),
                "exclude_block": i,

                "train_range_start": (accum_step, accum_step + train_slider_leng),
                "train_range_end": (3000, 2000),
            }
        )

        accum_step += step_size + (1 if i < rem else 0)
    return results


def get_n_fold_splits(train_leng, validation_leng, folds_count=5):
    results = []

    step_size = (train_leng - validation_leng) // (folds_count - 1)

    rem = (train_leng - validation_leng) % (folds_count - 1)

    accum_step = 0
    for i in range(folds_count):
        results.append(
            {
                "exclude_train_range_start": (accum_step, accum_step + validation_leng),
                "exclude_train_range_end": (3000, 2000),

                "exclude_range": (accum_step, accum_step + validation_leng),
                "exclude_range_end": (0, 0),
                "exclude_block": i,

                "train_range_start": (0, accum_step),
                "train_range_end": (accum_step + validation_leng, train_leng),
            }
        )
        accum_step += step_size + (1 if i < rem else 0)
    return results


def get_out_of_sample_fold_split(train_leng, validation_leng):
    results = [
        {
            "exclude_train_range_start": (train_leng - validation_leng, train_leng),
            "exclude_train_range_end": (3000, 2000),

            "exclude_range": (train_leng - validation_leng, train_leng),
            "exclude_range_end": (0, 0),
            "exclude_block": 0,

            "train_range_start": (0, train_leng - validation_leng),
            "train_range_end": (3000, 2000),
        }
    ]

    return results


def get_n_widening_fold_splits(train_leng, folds_count=5, initial_length=0, validation_leng=None):
    results = []

    # take biggest step s.t. we don't overstep the boundary
    step_size = (train_leng - validation_leng - initial_length) // (folds_count - 1)
    rem = (train_leng - validation_leng - initial_length) % (folds_count - 1)

    accum_step = initial_length
    for i in range(folds_count):
        results.append(
            {
                "exclude_train_range_start": (accum_step, train_leng),
                "exclude_train_range_end": (3000, 2000),

                "exclude_range": (accum_step, accum_step + validation_leng),
                "exclude_range_end": (0, 0),

                "train_range_start": (0, accum_step),
                "train_range_end": (3000, 2000),

                "exclude_block": i,
            }
        )
        accum_step += step_size + (1 if i < rem else 0)
    return results
