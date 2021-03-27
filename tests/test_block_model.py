""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge
import numpy as np
import pandas as pd

v = pd.DataFrame([['ING', 'NL'],
                 ['ABN', 'NL'],
                 ['UNI', 'IT'],
                 ['BNL', 'IT']],
                 columns=['name', 'country'])

e = pd.DataFrame([['ING', 'ABN', 1e6],
                 ['ABN', 'UNI', 2.3e7],
                 ['BNL', 'UNI', 7e5],
                 ['UNI', 'BNL', 3e3]],
                 columns=['creditor', 'debtor', 'value'])

g = ge.Graph(v, e, v_id='name', src='creditor', dst='debtor',
             weight='value', v_group='country')

# Define graph marginals to check computation
out_strength = np.rec.array([(0, 0, 1e6),
                             (1, 1, 2.3e7),
                             (2, 1, 3e3),
                             (3, 1, 7e5)],
                            dtype=[('id', np.uint8),
                                   ('group', np.uint8),
                                   ('value', np.float64)])

in_strength = np.rec.array([(1, 0, 1e6),
                            (2, 0, 2.3e7),
                            (2, 1, 7e5),
                            (3, 1, 3e3)],
                           dtype=[('id', np.uint8),
                                  ('group', np.uint8),
                                  ('value', np.float64)])

num_vertices = 4
num_edges = 4
num_groups = 2
group_dict = {1: 0, 2: 1, 0: 0, 3: 1}
z = 1


class TestBlockFitnessModel():
    def test_issubclass(self):
        """ Check that the block model is a graph ensemble."""
        model = ge.BlockFitnessModel(g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        """ Check that the block model can be correctly initialized from
        parameters directly.
        """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     num_edges=num_edges)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_groups == num_groups)
        assert np.all(model.group_dict == np.array([0, 0, 1, 1]))
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_g(self):
        """ Check that the block model can be correctly initialized from a
        graph.
        """
        model = ge.BlockFitnessModel(g)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_groups == num_groups)
        assert np.all(model.group_dict == np.array([0, 0, 1, 1]))
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_z(self):
        """ Check that the block model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     z=z)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.z == z)
        assert np.all(model.num_groups == num_groups)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.num_edges,
                                   num_edges,
                                   rtol=1e-6)

    def test_solver_newton(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.BlockFitnessModel(g)
        model.fit(method="newton")
        exp_num_edges = model.expected_num_edges()
        np.testing.assert_allclose(num_edges, exp_num_edges,
                                   atol=1e-8, rtol=0)

    def test_solver_fixed_point(self):
        """ Check that the fixed-point solver is fitting the z parameters
        correctly.
        """
        model = ge.BlockFitnessModel(g)
        model.fit(method="fixed-point", max_iter=100000, xtol=1e-5)
        exp_num_edges = model.expected_num_edges()
        np.testing.assert_allclose(num_edges, exp_num_edges,
                                   atol=1e-4, rtol=0)

    def test_sampling(self):
        """ Check that properties of the sample correspond to the ensemble.
        """
        model = ge.BlockFitnessModel(num_vertices=num_vertices,
                                     num_groups=num_groups,
                                     group_dict=group_dict,
                                     out_strength=out_strength,
                                     in_strength=in_strength,
                                     z=z)

        samples = 10000
        avg = 0
        for i in range(samples):
            sample = model.sample()
            avg += sample.num_edges
            assert sample.num_groups == num_groups
            assert sample.num_vertices == num_vertices

        avg = avg / samples
        np.testing.assert_isclose(avg, num_edges, atol=1e-1, rtol=0)
