"""Independent checks of sparse realization loss curve statistics."""
import numpy
from openquake.calculators import base
from openquake.calculators.extract import extract
from openquake.calculators.tests import CalculatorTestCase
from openquake.qa_tests_data.event_based_risk import case_01
from openquake.risklib import scientific


class SparseRealizationTest(CalculatorTestCase):
    def test_sparse_curve_mean(self):
        self.run_calc(case_01.__file__, 'job_ins.ini',
                      concurrent_tasks='1', collect_rlzs='false')
        ds = self.calc.datastore
        oq = ds['oqparam']
        weights = base.get_weights(oq, ds)
        numpy.testing.assert_allclose(weights, [.5, .5])
        counts = numpy.bincount(ds['events']['rlz_id'], minlength=2)
        print('OQ11 events per realization:', counts)
        assert (counts > 0).all()
        agg_id = list(base.general.decode(ds['agg_keys'][:])).index('A\tRM')
        df = ds.read_df('aggcurves')
        df = df[(df.agg_id == agg_id) &
                (df.loss_id == scientific.LOSSID['nonstructural'])]
        print('OQ11 native sparse curve rows:\n', df.to_string(index=False))
        assert set(df.rlz_id) == {0}
        common = ('&loss_type=nonstructural&absolute=1'
                  '&policy=A&taxonomy=RM')
        individual = extract(ds, 'agg_curves?kind=rlzs' + common)
        stored = extract(ds, 'agg_curves?kind=stats' + common)
        full = numpy.asarray(individual.array).reshape(2, -1)
        mean = numpy.asarray(stored.array).reshape(-1)
        assert (full[1] == 0).all()
        assert (full[0] > 0).any()
        expected = weights @ full
        print('OQ11 individual:', full)
        print('OQ11 stored:', mean)
        print('OQ11 independent weighted mean:', expected)
        numpy.testing.assert_allclose(mean, expected, rtol=1e-6)
