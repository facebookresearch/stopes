# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

K = tp.TypeVar("K")
V = tp.TypeVar("V")

log = logging.getLogger("generational_cache")


class GenerationalCache(tp.Generic[K, V]):
    """Multi-generation cache.

    Items are inserted in the top layer.
    Items move down one layer once they have been read at least some amount of times.
    When the cache is full, we reset the top layer.

    Thresholds indicates the number of hits before being moved down.
    """

    def __init__(self, thresholds: tp.List[int] = [10, 1], max_entry: int = 1_000_000):
        self.thresholds = thresholds
        self.generations: tp.List[tp.Dict[K, tp.Tuple[V, int]]] = [
            {} for _ in range(len(thresholds) + 1)
        ]
        self.n = 0
        self.max_entry = max_entry
        self.n_hits, self.n_queries = (0, 0)

    def get(self, key: K) -> tp.Optional[V]:
        self.n_queries += 1
        for i, g in enumerate(self.generations):
            val, count = g.get(key, (None, 0))  # type: ignore
            if val is not None:
                if i > 0:
                    count += 1
                    if count >= self.thresholds[i - 1]:
                        g.pop(key)
                        self.generations[i - 1][key] = (val, count)
                self.n_hits += 1
                return val
        return None

    def __setitem__(self, key, val) -> None:
        if self.n > self.max_entry:
            self.purge()
        self.n += 1
        self.generations[-1][key] = (val, 0)

    def __repr__(self) -> str:
        occupancy = self.n / self.max_entry
        hit_rate = self.n_hits / self.n_queries if self.n_queries else 0.0
        return f"{type(self).__name__}(gen={[len(g) for g in self.generations]}, total={self.n}, occupancy={occupancy:.2%}, hit_rate={hit_rate:.2%})"

    def purge(self) -> None:
        log.warning(f"Purging {self}")
        for g in reversed(self.generations):
            self.n -= len(g)
            g.clear()
            # Avoid the degenerate case where all items are in the second layer
            # and we need to purge on every insert.
            if self.n < self.max_entry / 2:
                return

    def reset_counts(self) -> None:
        """Reset counts but keep all cache entries."""
        g_base = self.generations[-1]
        for g in self.generations[:-1]:
            for key, (val, count) in g.items():
                g_base[key] = (val, 1)
            g.clear()

    def map_batches(
        self,
        fn: tp.Callable[[tp.List[K]], tp.List[V]],
        it: tp.Iterable[K],
        batch_size: int,
        fn_dict: dict = None,
    ):

        return map_batches(fn, it, batch_size, cache=self, fn_dict=fn_dict)


def map_batches(
    fn: tp.Callable[[tp.List[K]], tp.List[V]],
    it: tp.Iterable[K],
    batch_size: int,
    cache: GenerationalCache[K, V] = None,
    fn_dict: dict = None,
):
    # When translating a website we will see some sentences a lot of times,
    # typically header/footer/menus. So we use a generational cache.
    cache = cache or GenerationalCache([10, 1])
    batch = []
    batch_with_results = []

    def handle_batch(batch, batch_with_results) -> tp.Iterator[V]:
        if batch:
            new_results = fn(batch, fn_dict) if fn_dict is not None else fn(batch)
        else:
            new_results = []
        assert len(batch) == len(new_results)
        for x, res in zip(batch, new_results):
            cache[x] = res
        j = 0
        for (x, res) in batch_with_results:
            if res is None:
                res = new_results[j]
                j += 1
            yield res

        assert j == len(new_results)
        batch_with_results.clear()
        batch.clear()

    for x in it:
        res = cache.get(x)
        batch_with_results.append((x, res))
        if res is None:
            batch.append(x)

        if len(batch) >= batch_size:
            yield from handle_batch(batch, batch_with_results)

    if batch_with_results:
        yield from handle_batch(batch, batch_with_results)
    log.warning(f"Cache {cache}")


def test_map_batches():
    def fn(xs):
        return [x + 1 for x in xs]

    assert list(map_batches(fn, range(10), 3)) == list(range(1, 11))
    assert (
        list(map_batches(fn, [i % 10 for i in range(20)], 3)) == list(range(1, 11)) * 2
    )
    assert list(map_batches(fn, range(100), 16)) == list(range(1, 101))
