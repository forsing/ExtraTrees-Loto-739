#!/usr/bin/env python3
"""
ExtraTrees ideja na Loto 7/39.
Regresija numerickog opisa distribucije.
Random samo unutar ExtraTrees, uz fiksno seme.
Kombinacija se bira deterministicki.

CSV: sedam celih brojeva po redu, bez zaglavlja,
redosled od najstarijeg ka najnovijem izvlacenju.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


DEFAULTS = (
    "/data/loto7_4702_k82.csv",
    "/data/loto7_4702_k82_loto_2973.csv",
    "/data/loto7_4702_k82_loto_plus_1729.csv",
)

SEED = 42
TREES = 1000
DEPTH = 15
TRAIN = 384
VALIDATION = 48
TEST = 64
REFIT = 16

LAGS = (1, 2, 3, 4, 5, 8, 10)
CONFIGS = (1, 2, 4)

PI, PJ = np.triu_indices(7, 1)


def read_csv(path):
    rows = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        for line, cells in enumerate(csv.reader(f), 1):
            if not cells or all(not c.strip() for c in cells):
                continue

            try:
                row = tuple(int(c.strip()) for c in cells)
            except ValueError as exc:
                raise ValueError(
                    f"{path}: red {line} nije sedam celih brojeva"
                ) from exc

            if (
                len(row) != 7
                or len(set(row)) != 7
                or min(row) < 1
                or max(row) > 39
            ):
                raise ValueError(
                    f"{path}: neispravan red {line}: {row}"
                )

            rows.append(tuple(sorted(row)))

    minimum = TRAIN + max(LAGS) + VALIDATION + TEST

    if len(rows) < minimum:
        raise ValueError(
            f"{path}: potrebno najmanje {minimum} izvlacenja"
        )

    return np.asarray(rows, dtype=np.int16)


def distributions(rows):
    c = np.atleast_2d(rows).astype(float)

    # CDF unutar svakog pojedinacnog izvlacenja.
    # Nema tabela ukupnih ili recentnih frekvencija brojeva.
    cdf = (
        c[:, :, None] <= np.arange(1, 40)
    ).mean(axis=1)

    # Distribucija svih 21 rastojanja medju brojevima.
    # Ovo nije predstavljanje pomocu osam razmaka.
    distances = c[:, PJ] - c[:, PI]

    pair_cdf = np.column_stack([
        (distances <= boundary).mean(axis=1)
        for boundary in (2, 5, 10, 20, 30)
    ])

    odd = (c % 2 == 1).mean(axis=1)
    low = (c <= 19).mean(axis=1)
    span = (c[:, -1] - c[:, 0]) / 38

    # Osobine za ucenje, bez propisanih ciljnih vrednosti.
    return np.column_stack((
        cdf,
        pair_cdf,
        odd,
        low,
        span,
    ))


def build_inputs(d):
    n = len(d)

    x = np.zeros((
        n + 1,
        len(LAGS) * d.shape[1],
    ))

    # X[t] uvek koristi iskljucivo zavrsena izvlacenja.
    # Isti postupak vazi za trening, test i NEXT.
    for t in range(max(LAGS), n + 1):
        x[t] = np.concatenate([
            d[t - lag] for lag in LAGS
        ])

    return x


class Predictor:
    def __init__(
        self,
        rows,
        x,
        d,
        t,
        config,
        trees=TREES,
    ):
        start = max(max(LAGS), t - TRAIN)

        self.model = ExtraTreesRegressor(
            n_estimators=trees,
            max_depth=DEPTH,
            min_samples_leaf=config,
            criterion="squared_error",
            # Originalni klasifikator podrazumeva sqrt.
            max_features="sqrt",
            bootstrap=False,
            random_state=SEED,
            n_jobs=1,
        )

        # Trening zavrsava pre izvlacenja t.
        self.model.fit(
            x[start:t],
            d[start:t],
        )

    def distribution(self, query):
        predicted = self.model.predict(
            np.asarray(query).reshape(1, -1)
        )[0]

        cdf = predicted[:39]

        if (
            not np.all(np.isfinite(cdf))
            or not np.isclose(cdf[-1], 1)
        ):
            raise RuntimeError(
                "Neispravna regresiona distribucija"
            )

        # Razlika CDF-a daje modelsku ocenu ukljucivanja.
        mass = 7 * np.diff(np.r_[0.0, cdf])

        if (
            mass.min() < -1e-10
            or not np.isclose(mass.sum(), 7)
        ):
            raise RuntimeError(
                "CDF regresora nije ispravan"
            )

        # Iskljucivo zastita od sitne numericke greske.
        return np.maximum(mass, 0)

    def next(self, query, previous):
        mass = self.distribution(query)

        # Zadrzan originalni top-k postupak.
        # Bez random izbora i bez filtera oblika kombinacije.
        order = np.argsort(
            -mass,
            kind="stable",
        )[:7]

        result = tuple(sorted(
            int(i) + 1 for i in order
        ))

        if (
            len(result) != 7
            or len(set(result)) != 7
        ):
            raise RuntimeError(
                "Neispravna kombinacija"
            )

        return result


def hits(a, b):
    return len(set(a) & set(b))


def evaluate(
    rows,
    x,
    d,
    start,
    end,
    config,
):
    predictions = []
    model = None

    for t in range(start, end):
        if (
            model is None
            or (t - start) % REFIT == 0
        ):
            model = Predictor(
                rows,
                x,
                d,
                t,
                config,
            )

        predictions.append(
            model.next(x[t], rows[t - 1])
        )

    actual = np.array([
        hits(p, rows[t])
        for t, p in zip(
            range(start, end),
            predictions,
        )
    ])

    control = np.array([
        hits(rows[t - 1], rows[t])
        for t in range(start, end)
    ])

    return actual, control, predictions


def diagnostics(predictions):
    runs = []
    groups = 0

    for c in predictions:
        longest = current = 1

        for a, b in zip(c, c[1:]):
            current = (
                current + 1 if b == a + 1 else 1
            )
            longest = max(longest, current)

        runs.append(longest)

        low = [n for n in c if n <= 19]
        high = [n for n in c if n >= 20]

        # Samo izvestaj. Nikada filter, kazna ili uslov izbora.
        if len(low) >= 2 and len(high) >= 2:
            groups += int(
                low[-1] - low[0] <= 7
                and high[-1] - high[0] <= 7
                and high[0] - low[-1] >= 10
            )

    return (
        max(runs),
        sum(r >= 5 for r in runs),
        groups,
    )


def run(path):
    rows = read_csv(path)
    n = len(rows)

    d = distributions(rows)
    x = build_inputs(d)

    test_start = n - TEST
    val_start = test_start - VALIDATION

    print("\n" + str(Path(path)), flush=True)

    print(
        f"Redova: {n}; poslednja: "
        + ",".join(map(str, rows[-1])),
        flush=True,
    )

    validation = []

    for config in CONFIGS:
        scores, _, _ = evaluate(
            rows,
            x,
            d,
            val_start,
            test_start,
            config,
        )

        validation.append(float(scores.mean()))

        print(
            f"Validacija {config}: "
            f"{scores.mean():.4f}",
            flush=True,
        )

    index = max(
        range(len(CONFIGS)),
        key=lambda i: (validation[i], -i),
    )

    config = CONFIGS[index]

    scores, control, predictions = evaluate(
        rows,
        x,
        d,
        test_start,
        n,
        config,
    )

    print(
        "Parametri zamrznuti posle validacije: "
        f"min_samples_leaf={config}"
    )

    print(
        f"Test {TEST}: model={scores.mean():.4f}; "
        "Kontrola: prethodno izvuceno kolo="
        f"{control.mean():.4f}"
    )

    print(
        f"Razlika={(scores - control).mean():+.4f}; "
        f"3+={int((scores >= 3).sum())}/{TEST}; "
        f"4+={int((scores >= 4).sum())}/{TEST}"
    )

    print(
        "Distribucija pogodaka 0..7:",
        np.bincount(
            scores,
            minlength=8,
        ).tolist(),
    )

    print(
        f"Razlicitih test predikcija: "
        f"{len(set(predictions))}/{TEST}"
    )

    replay = sum(
        any(tuple(r) == p for r in rows[:t])
        for t, p in zip(
            range(test_start, n),
            predictions,
        )
    )

    print(
        f"Istorijska ponavljanja u testu: "
        f"{replay}/{TEST}"
    )

    run_length, long_runs, groups = diagnostics(
        predictions
    )

    print(
        f"Najduzi niz: {run_length}; "
        f"niz 5+: {long_runs}; "
        "dve zbijene polovine "
        f"(opis, bez filtera): {groups}"
    )

    model = Predictor(
        rows,
        x,
        d,
        n,
        config,
    )

    result = model.next(x[n], rows[-1])
    again = model.next(x[n], rows[-1])

    if result != again:
        raise RuntimeError(
            "Izbor kombinacije nije ponovljiv"
        )

    print(
        f"NEXT red {n + 1}: "
        + ",".join(map(str, result))
    )

    print(
        "NEXT vec postoji u istoriji:",
        any(tuple(r) == result for r in rows),
    )

    print(
        "NEXT najduzi uzastopni niz:",
        diagnostics([result])[0],
    )

    if scores.mean() <= control.mean():
        print(
            "Na ovom testu model nije imao "
            "bolji prosek od kontrole."
        )

    return result


def self_check():
    rows = np.array([
        sorted({
            (3 * i + 5 * j) % 39 + 1
            for j in range(7)
        })
        for i in range(100)
    ], dtype=np.int16)

    d = distributions(rows)
    x = build_inputs(d)

    changed = rows.copy()
    changed[70:] = rows[0]

    changed_d = distributions(changed)
    x_changed = build_inputs(changed_d)

    # Promena izvlacenja 70 i kasnijih ne menja X[70].
    assert np.array_equal(
        x[:71],
        x_changed[:71],
    )

    assert np.array_equal(
        x[70],
        build_inputs(d[:70])[70],
    )

    first = Predictor(
        rows,
        x,
        d,
        70,
        CONFIGS[0],
        trees=32,
    )

    second = Predictor(
        changed,
        x_changed,
        changed_d,
        70,
        CONFIGS[0],
        trees=32,
    )

    p = first.distribution(x[70])
    p2 = second.distribution(x_changed[70])

    assert np.array_equal(p, p2)
    assert np.isclose(p.sum(), 7)

    assert (
        first.next(x[70], rows[69])
        == second.next(
            x_changed[70],
            changed[69],
        )
    )


def main():
    self_check()

    print(
        f"ExtraTreesRegressor: {TREES} stabala; "
        f"dubina={DEPTH}; seme={SEED}."
    )

    print(
        "Random samo u stablima. "
        "Izbor kombinacije je deterministicki."
    )

    print(
        "Bez nametnutih filtera oblika kombinacije."
    )

    outputs = []

    # CSV-i se nikada ne spajaju.
    for path in (sys.argv[1:] or DEFAULTS):
        try:
            outputs.append((path, run(path)))
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    # Jednake izlaze prijavljuje, ne menja ih vestacki.
    for i, (left, a) in enumerate(outputs):
        for right, b in outputs[i + 1:]:
            if a == b:
                print(
                    f"ISTA NEXT: {Path(left).name} / "
                    f"{Path(right).name}: {a}"
                )


if __name__ == "__main__":
    main()



"""
ExtraTreesRegressor: 1000 stabala; dubina=15; seme=42.
Random samo u stablima. Izbor kombinacije je deterministicki.
Bez nametnutih filtera oblika kombinacije.

/data/loto7_4702_k82.csv
Redova: 4702; poslednja: 8,10,13,21,22,23,32
Validacija 1: 1.3542
Validacija 2: 1.3125
Validacija 4: 1.2083
Parametri zamrznuti posle validacije: min_samples_leaf=1
Test 64: model=1.3281; Kontrola: prethodno izvuceno kolo=1.1719
Razlika=+0.1562; 3+=6/64; 4+=0/64
Distribucija pogodaka 0..7: [10, 29, 19, 6, 0, 0, 0, 0]
Razlicitih test predikcija: 64/64
Istorijska ponavljanja u testu: 0/64
Najduzi niz: 3; niz 5+: 0; dve zbijene polovine (opis, bez filtera): 1
NEXT red 4703: 8,x,25,y,34,z,39
NEXT vec postoji u istoriji: False
NEXT najduzi uzastopni niz: 3



/data/loto7_4702_k82_loto_2973.csv
Redova: 2973; poslednja: 1,2,8,26,29,35,39
Validacija 1: 1.1042
Validacija 2: 1.1250
Validacija 4: 1.0625
Parametri zamrznuti posle validacije: min_samples_leaf=2
Test 64: model=1.2188; Kontrola: prethodno izvuceno kolo=1.3281
Razlika=-0.1094; 3+=5/64; 4+=0/64
Distribucija pogodaka 0..7: [14, 27, 18, 5, 0, 0, 0, 0]
Razlicitih test predikcija: 64/64
Istorijska ponavljanja u testu: 0/64
Najduzi niz: 5; niz 5+: 1; dve zbijene polovine (opis, bez filtera): 1
NEXT red 2974: 22,x,25,y,33,z,38
NEXT vec postoji u istoriji: False
NEXT najduzi uzastopni niz: 2
Na ovom testu model nije imao bolji prosek od kontrole.



/data/loto7_4702_k82_loto_plus_1729.csv
Redova: 1729; poslednja: 8,10,13,21,22,23,32
Validacija 1: 1.3542
Validacija 2: 1.3333
Validacija 4: 1.2917
Parametri zamrznuti posle validacije: min_samples_leaf=1
Test 64: model=1.2969; Kontrola: prethodno izvuceno kolo=1.2656
Razlika=+0.0312; 3+=5/64; 4+=1/64
Distribucija pogodaka 0..7: [11, 29, 19, 4, 1, 0, 0, 0]
Razlicitih test predikcija: 64/64
Istorijska ponavljanja u testu: 0/64
Najduzi niz: 4; niz 5+: 0; dve zbijene polovine (opis, bez filtera): 1
NEXT red 1730: 2,x,9,y,18,z,34
NEXT vec postoji u istoriji: False
NEXT najduzi uzastopni niz: 2
"""



"""
ExtraTreesRegressor koristi nasumične pragove pri treniranju. 
Fiksno seme daje ponovljive rezultate, ali taj korak ostaje random.

ExtraTrees sa 1.000 stabala i originalni top-k izbor. 
Regresor uči numerički opis distribucije prethodnih izvlačenja. 
Slučajnost postoji samo unutar stabala, uz seme 42.

Provereno na sva tri CSV-a:

CSV	    Prosek pogodaka, 64 kola	Kontrola: prethodno izvučeno kolo
4702	1,3281	                    1,1719
2973	1,2188	                    1,3281
1729	1,2969	                    1,2656

Sve test predikcije bile su različite unutar svakog CSV-a. 
Model nije potpuno rešio grupisanje: 
na Loto testu jednom je proizveo pet uzastopnih brojeva; 
njegova sadašnja NEXT kombinacija je cela u gornjoj polovini. 
Nisam to sakrio zabranama ili naknadnim prepravljanjem.
"""
