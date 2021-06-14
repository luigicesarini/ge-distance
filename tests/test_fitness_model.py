""" Test the fitness model class on simple sample graph. """
import graph_ensembles as ge
import numpy as np
import pandas as pd
import pytest

v = pd.DataFrame([['ING', 'NL'],
                 ['ABN', 'NL'],
                 ['BNP', 'FR'],
                 ['BNP', 'IT']],
                 columns=['name', 'country'])

e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6],
                 ['BNP', 'FR', 'ABN', 'NL', 2.3e7],
                 ['BNP', 'IT', 'ABN', 'NL', 7e5 + 3e3],
                 ['ABN', 'NL', 'BNP', 'FR', 1e4],
                 ['ABN', 'NL', 'ING', 'NL', 4e5]],
                 columns=['creditor', 'c_country',
                          'debtor', 'd_country',
                          'value'])

e_l = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6, 'interbank', False],
                   ['BNP', 'FR', 'ABN', 'NL', 2.3e7, 'external', False],
                   ['BNP', 'IT', 'ABN', 'NL', 7e5, 'interbank', True],
                   ['BNP', 'IT', 'ABN', 'NL', 3e3, 'interbank', False],
                   ['ABN', 'NL', 'BNP', 'FR', 1e4, 'interbank', False],
                   ['ABN', 'NL', 'ING', 'NL', 4e5, 'external', True]],
                   columns=['creditor', 'c_country',
                            'debtor', 'd_country',
                            'value', 'type', 'EUR'])

g = ge.Graph(v, e, v_id=['name', 'country'],
             src=['creditor', 'c_country'],
             dst=['debtor', 'd_country'],
             weight='value')

g_l = ge.Graph(v, e_l, v_id=['name', 'country'],
               src=['creditor', 'c_country'],
               dst=['debtor', 'd_country'],
               edge_label=['type', 'EUR'],
               weight='value')

# Define graph marginals to check computation
out_strength = np.array([1e6, 1e4 + 4e5, 2.3e7, 7e5 + 3e3],
                        dtype=np.float64)

in_strength = np.array([4e5, 1e6 + 3e3 + 2.3e7 + 7e5, 1e4, 0],
                       dtype=np.float64)

num_vertices = 4
num_edges = 5
z = 4.291803e-12
z_inv = 2.913604e-12


class TestFitnessModelInit():
    def test_issubclass(self):
        """ Check that the model is a graph ensemble."""
        model = ge.FitnessModel(g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        model = ge.FitnessModel(g)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

        model = ge.FitnessModel(g_l)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_param(self):
        """ Check that the model can be correctly initialized from
        parameters directly.
        """
        model = ge.FitnessModel(num_vertices=num_vertices,
                                out_strength=out_strength,
                                in_strength=in_strength,
                                num_edges=num_edges)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_vertices == num_vertices)

    def test_model_init_z(self):
        """ Check that the model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.FitnessModel(num_vertices=num_vertices,
                                out_strength=out_strength,
                                in_strength=in_strength,
                                param=z)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.param == z)
        assert np.all(model.num_vertices == num_vertices)
        np.testing.assert_allclose(model.num_edges,
                                   num_edges,
                                   rtol=1e-5)

    def test_model_wrong_init(self):
        """ Check that the model raises exceptions for wrong inputs."""
        msg = 'First argument passed must be a WeightedGraph.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel('df', 234, out_strength)

        msg = 'Unnamed arguments other than the Graph have been ignored.'
        with pytest.warns(UserWarning, match=msg):
            ge.FitnessModel(g, 'df', 234, out_strength)

        msg = 'Illegal argument passed: num_nodes'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_nodes=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            num_edges=num_edges)

    def test_wrong_num_vertices(self):
        """ Check that wrong initialization of num_vertices results in an
        error.
        """
        msg = 'Number of vertices not set.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(out_strength=out_strength,
                            in_strength=in_strength,
                            num_edges=num_edges)

        msg = 'Number of vertices must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=np.array([1, 2]),
                            out_strength=out_strength,
                            in_strength=in_strength,
                            num_edges=num_edges)

        msg = 'Number of vertices must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=-3,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            num_edges=num_edges)

    def test_wrong_strengths(self):
        """ Check that wrong initialization of strengths results in an error.
        """
        msg = 'out_strength not set.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            in_strength=in_strength,
                            num_edges=num_edges)

        msg = 'in_strength not set.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            num_edges=num_edges)

        msg = ("Out strength must be a numpy array of length " +
               str(num_vertices))
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=1,
                            in_strength=in_strength,
                            num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength[0:2],
                            in_strength=in_strength,
                            num_edges=num_edges)

        msg = ("In strength must be a numpy array of length " +
               str(num_vertices))
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=2,
                            num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength[0:2],
                            num_edges=num_edges)

        msg = "Out strength must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=-out_strength,
                            in_strength=in_strength,
                            num_edges=num_edges)

        msg = "In strength must contain positive values only."
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=-in_strength,
                            num_edges=num_edges)

        msg = "Sums of strengths do not match."
        with pytest.raises(AssertionError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength + 1,
                            in_strength=in_strength,
                            num_edges=num_edges)

    def test_wrong_num_edges(self):
        """ Check that wrong initialization of num_edges results in an error.
        """
        msg = 'Number of edges must be a number.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            num_edges=np.array([1, 2]))

        msg = 'Number of edges must be a number.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            num_edges='3')

        msg = 'Number of edges must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            num_edges=-324)

        msg = 'Either num_edges or param must be set.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength)

    def test_wrong_z(self):
        """ Check that the passed z adheres to format.
        """
        msg = 'The FitnessModel requires one parameter.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            param=np.array([0, 1]))

        msg = ('The FitnessModel with min degree correction requires two '
               'parameters.')
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            min_degree=True,
                            param=np.array([0]))

        msg = 'Parameters must be numeric.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            param='0')

        msg = 'Parameters must be positive.'
        with pytest.raises(ValueError, match=msg):
            ge.FitnessModel(num_vertices=num_vertices,
                            out_strength=out_strength,
                            in_strength=in_strength,
                            param=-1)


class TestFitnessModelFit():
    def test_solver_newton(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.FitnessModel(g)
        model.fit(method="newton", tol=1e-6)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_invariant(self):
        """ Check that the newton solver is fitting the z parameters
        correctly for the invariant case. """
        model = ge.FitnessModel(g, scale_invariant=True)
        model.fit(method="newton", tol=1e-6)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_inv, model.param, atol=0, rtol=1e-6)

    def test_solver_fixed_point(self):
        """ Check that the fixed-point solver is fitting the z parameters
        correctly.
        """
        model = ge.FitnessModel(g)
        model.fit(method="fixed-point", max_iter=100, xtol=1e-5)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-4, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-4)

    def test_solver_min_degree(self):
        """ Check that the min_degree solver converges.
        """
        model = ge.FitnessModel(g, min_degree=True)
        model.fit(x0=np.array([1e-2, 0.07]), tol=1e-6, max_iter=500)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        model.expected_degrees()
        exp_d_out = model.exp_out_degree
        exp_d_in = model.exp_in_degree
        assert np.all(exp_d_out[exp_d_out != 0] >= 1 - 1e-5)
        assert np.all(exp_d_in[exp_d_in != 0] >= 1 - 1e-5)

    def test_solver_with_init(self):
        """ Check that it works with a given initial condition.
        """
        model = ge.FitnessModel(g)
        model.fit(x0=1e-14)
        model.expected_num_edges()
        np.testing.assert_allclose(num_edges, model.exp_num_edges,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z, model.param, atol=0, rtol=1e-6)

    def test_solver_with_wrong_init(self):
        """ Check that it raises an error with a negative initial condition.
        """
        model = ge.FitnessModel(g)
        msg = "x0 must be positive."
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=-1)

        msg = 'The FitnessModel requires one parameter.'
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=np.array([0, 1]))

        msg = ('The FitnessModel with min degree correction requires two '
               'parameters.')
        model = ge.FitnessModel(g, min_degree=True)
        with pytest.raises(ValueError, match=msg):
            model.fit(x0=np.array([0]))

        msg = 'x0 must be numeric.'
        with pytest.raises(ValueError, match=msg):
            model.fit(x0='hi')

    def test_wrong_method(self):
        """ Check that wrong methods names return an error.
        """
        model = ge.FitnessModel(g)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")

    def test_method_incompatibility(self):
        """ Check that an error is raised when trying to use the wrong method.
        """
        model = ge.FitnessModel(g, scale_invariant=True)
        msg = ('Fixed point solver not supported for scale '
               'invariant functional.')
        with pytest.raises(Exception, match=msg):
            model.fit(method="fixed-point", max_iter=100, xtol=1e-5)

        # model = ge.FitnessModel(g, min_degree=True)
        # msg = ('Method not recognised for solver with min degree '
        #        'constraint, using default SLSQP.')
        # with pytest.warns(UserWarning, match=msg):
        #     model.fit(method="newton")

        msg = 'Cannot constrain min degree in scale invariant model.'
        with pytest.raises(Exception, match=msg):
            model = ge.FitnessModel(g, scale_invariant=True, min_degree=True)


class TestFitnessModelMeasures():
    def test_exp_n_edges(self):
        """ Check expected edges is correct. """
        model = ge.FitnessModel(num_vertices=num_vertices,
                                out_strength=out_strength,
                                in_strength=in_strength,
                                param=z)
        model.expected_num_edges()
        np.testing.assert_allclose(model.exp_num_edges,
                                   num_edges,
                                   rtol=1e-5)

    def test_exp_out_degree(self):
        """ Check expected d_out is correct. """
        model = ge.FitnessModel(num_vertices=num_vertices,
                                out_strength=out_strength,
                                in_strength=in_strength,
                                param=z)
        model.expected_degrees()
        d_out = model.exp_out_degree
        np.testing.assert_allclose(
            d_out, np.array([1.031808, 0.430388, 1.974889, 1.562915]),
            rtol=1e-5)
        np.testing.assert_allclose(num_edges, np.sum(d_out))

    def test_exp_in_degree(self):
        """ Check expected d_out is correct. """
        model = ge.FitnessModel(num_vertices=num_vertices,
                                out_strength=out_strength,
                                in_strength=in_strength,
                                param=z)
        model.expected_degrees()
        d_in = model.exp_in_degree
        np.testing.assert_allclose(
            d_in, np.array([1.935262, 2.977007, 0.087732, 0]), rtol=1e-5)
        np.testing.assert_allclose(num_edges, np.sum(d_in))


class TestFitnessModelSample():
    def test_sampling(self):
        """ Check that properties of the sample correspond to ensemble.
        """
        model = ge.FitnessModel(num_vertices=num_vertices,
                                out_strength=out_strength,
                                in_strength=in_strength,
                                param=z)

        samples = 10000
        s_n_e = np.empty(samples)
        for i in range(samples):
            sample = model.sample()
            s_n_e[i] = sample.num_edges

        s_n_e = np.average(s_n_e, axis=0)
        np.testing.assert_allclose(s_n_e, num_edges, atol=1e-1, rtol=0)
