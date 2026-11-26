"""Research-only regression for sparse realization loss curves."""
from functools import partial
from types import SimpleNamespace
from unittest import mock

import numpy
import pandas

from openquake.calculators import base
from openquake.calculators.extract import extract
from openquake.calculators.post_risk import save_curve_stats
from openquake.calculators.tests import CalculatorTestCase
from openquake.qa_tests_data.event_based_risk import case_01
from openquake.risklib import scientific
from openquake.hazardlib import stats


class SparseRealizationTestCase(CalculatorTestCase):
    def test_oq11_native(self):
        self.run_calc(case_01.__file__, 'job_ins.ini',
                      collect_rlzs='false')
        ds = self.calc.datastore
        oq = ds['oqparam']
        weights = base.get_weights(oq, ds)
        numpy.testing.assert_allclose(weights, [.5, .5])
        counts = numpy.bincount(ds['events']['rlz_id'], minlength=2)
        assert (counts > 0).all(), counts
        curves = ds.read_df('aggcurves')
        stats = extract(ds, 'risk_stats/aggcurves')
        tags = [key.decode() if isinstance(key, bytes) else key
                for key in ds['agg_keys'][:]]
        agg_id = tags.index('A\tRM')
        control_id = tags.index('A\tRC')
        periods = curves.return_period.unique()
        stat_idx = list(oq.hazard_stats()).index('mean')
        failures = []
        sparse_nonzero = 0
        for group_id in (agg_id, control_id):
            for lt in ('nonstructural', 'structural',
                       'structural+nonstructural_ins'):
                lid = scientific.LOSSID[lt]
                data = curves[(curves.agg_id == group_id) &
                              (curves.loss_id == lid)]
                stored = ds['agg_curves-stats/' + lt][group_id, stat_idx]
                for pi, rp in enumerate(periods):
                    rows = data[data.return_period == rp]
                    if rows.empty:
                        continue
                    vals = numpy.zeros(len(weights))
                    vals[rows.rlz_id.to_numpy()] = rows.loss.to_numpy()
                    if not numpy.isfinite(vals).all():
                        continue
                    expected = float(numpy.dot(vals, weights))
                    got = float(stored[pi, 0])
                    sel = stats[(stats.agg_id == group_id) &
                                (stats.loss_type == lt) &
                                (stats.return_period == rp) &
                                (stats.stat == 'mean')]
                    assert len(sel) == 1
                    exported = float(sel.loss.iloc[0])
                    numpy.testing.assert_allclose(
                        exported, expected, rtol=1e-6, atol=1e-7)
                    if len(rows) < len(weights) and expected > 0:
                        sparse_nonzero += 1
                    print(f'OQ11 group={tags[group_id]!r} {lt} rp={rp} '
                          f'rows={len(rows)} expected={expected:.9g} '
                          f'stored={got:.9g} exported={exported:.9g}',
                          flush=True)
                    if not numpy.isclose(got, expected, rtol=1e-6,
                                         atol=1e-7):
                        failures.append((group_id, lt, rp, got, expected))
        assert sparse_nonzero, 'Fixture no longer has nonzero sparse curves'
        assert not failures, f'OQ11 stored mean discrepancies: {failures}'

def test_sparse_curve_statistics():
    # Missing rows are zeros; explicit NaNs are not missing rows.
    weights = numpy.array([.25, .75], numpy.float32)
    funcs = dict(mean=stats.mean_curve, std=stats.std_curve,
                 median=partial(stats.quantile_curve, .5))
    lid = scientific.LOSSID['structural']
    df = pandas.DataFrame(dict(
        agg_id=[0, 0, 0, 0], loss_id=[lid] * 4,
        return_period=[10, 20, 20, 30], rlz_id=[1, 0, 1, 1],
        loss=numpy.array([80, 20, 80, numpy.nan], numpy.float32)))
    oq = SimpleNamespace(
        loss_types=['structural'], ext_loss_types=['structural'],
        number_of_logic_tree_samples=0, hazard_stats=lambda: funcs)
    ds = mock.MagicMock()
    datasets = {'oqparam': oq, 'weights': weights,
                'exposure': mock.Mock()}
    ds.__getitem__.side_effect = datasets.__getitem__
    ds.read_df.return_value = df

    def create(name, dtype, shape):
        datasets[name] = numpy.zeros(shape, dtype)

    ds.create_dset.side_effect = create
    save_curve_stats(ds)
    out = datasets['agg_curves-stats/structural'][0, :, :, 0]
    numpy.testing.assert_allclose(out[:, :2], [
        [60, 65], [numpy.sqrt(1200), numpy.sqrt(675)], [80 / 3, 40]],
        rtol=1e-6)
    assert numpy.isnan(out[:, 2]).all()
