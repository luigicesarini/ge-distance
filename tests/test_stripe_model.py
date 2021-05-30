""" Test graph ensemble model classes on simple sample graph. """
import graph_ensembles as ge
import numpy as np
import pandas as pd
import pytest
import re

v = pd.DataFrame([['ING', 'NL'],
                 ['ABN', 'NL'],
                 ['BNP', 'FR'],
                 ['BNP', 'IT']],
                 columns=['name', 'country'])

e = pd.DataFrame([['ING', 'NL', 'ABN', 'NL', 1e6, 'interbank', False],
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
             edge_label=['type', 'EUR'],
             weight='value')

# Define graph marginals to check computation
out_strength = np.rec.array([(0, 0, 1e6),
                             (0, 1, 1e4),
                             (0, 3, 3e3),
                             (1, 2, 2.3e7),
                             (2, 3, 7e5),
                             (3, 1, 4e5)],
                            dtype=[('label', np.uint8),
                                   ('id', np.uint8),
                                   ('value', np.float64)])

in_strength = np.rec.array([(0, 1, 1e6 + 3e3),
                            (0, 2, 1e4),
                            (1, 1, 2.3e7),
                            (2, 1, 7e5),
                            (3, 0, 4e5)],
                           dtype=[('label', np.uint8),
                                  ('id', np.uint8),
                                  ('value', np.float64)])

num_vertices = 4
num_edges = 5
num_edges_label = np.array([3, 1, 1, 1])
num_labels = 4
z = 8.99e-10
z_label = np.array([1.826524e-09, 2.477713e-10, 2.674918e-07, 8.191937e-07])
z_inv = 1e-1
z_inv_lbl = np.array([7.749510e-10, 2.268431e-14, 2.448980e-11, 7.500000e-11])


class TestStripeFitnessModelInit():
    def test_issubclass(self):
        """ Check that the stripe model is a graph ensemble."""
        model = ge.StripeFitnessModel(g)
        assert isinstance(model, ge.GraphEnsemble)

    def test_model_init(self):
        """ Check that the stripe model can be correctly initialized from
        a graph object.
        """
        model = ge.StripeFitnessModel(g, per_label=False)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label

        model = ge.StripeFitnessModel(g, per_label=True)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label

    def test_model_init_param(self):
        """ Check that the stripe model can be correctly initialized from
        parameters directly.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      num_edges=num_edges)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label

        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      num_edges=num_edges_label)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.num_edges == num_edges_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label

    def test_model_init_z(self):
        """ Check that the stripe model can be correctly initialized with
        the z parameter instead of num_edges.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.z == z)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert not model.per_label
        np.testing.assert_allclose(model.num_edges,
                                   num_edges,
                                   rtol=1e-5)

        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)
        assert np.all(model.out_strength == out_strength)
        assert np.all(model.in_strength == in_strength)
        assert np.all(model.z == z_label)
        assert np.all(model.num_labels == num_labels)
        assert np.all(model.num_vertices == num_vertices)
        assert model.per_label
        np.testing.assert_allclose(model.num_edges,
                                   num_edges_label,
                                   rtol=1e-5)

    def test_model_wrong_init(self):
        """ Check that the stripe model raises exceptions for wrong inputs."""
        msg = 'First argument passed must be a WeightedLabelGraph.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel('df', 234, out_strength)

        msg = 'Unnamed arguments other than the Graph have been ignored.'
        with pytest.warns(UserWarning, match=msg):
            ge.StripeFitnessModel(g, 'df', 234, out_strength)

        msg = 'Illegal argument passed: num_nodes'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_nodes=num_vertices,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_wrong_num_vertices(self):
        """ Check that wrong initialization of num_vertices results in an
        error.
        """
        msg = 'Number of vertices not set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of vertices smaller than max id value in strengths.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=2,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of vertices must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=np.array([1, 2]),
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of vertices must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=-3,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_wrong_num_labels(self):
        """ Check that wrong initialization of num_labels results in an error.
        """
        msg = 'Number of labels not set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of labels smaller than max label value in strengths.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=2,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of labels must be an integer.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=np.array([1, 2]),
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'Number of labels must be a positive number.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=-5,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_wrong_strengths(self):
        """ Check that wrong initialization of num_edges results in an error.
        """
        msg = 'out_strength not set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = 'in_strength not set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  num_edges=num_edges)

        msg = re.escape("Out strength must be a rec array with columns: "
                        "('label', 'id', 'value')")
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=1,
                                  in_strength=in_strength,
                                  num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength.value,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = re.escape("In strength must be a rec array with columns: "
                        "('label', 'id', 'value')")
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=2,
                                  num_edges=num_edges)
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength.value,
                                  num_edges=num_edges)

        msg = "Sums of strengths per label do not match."
        tmp = out_strength.copy()
        tmp.value[0] = tmp.value[0] + 1
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        msg = "Storing zeros in the strengths leads to inefficient code."
        tmp = out_strength.copy()
        tmp.resize(len(tmp) + 1)
        tmp[-1] = ((1, 2, 0))
        with pytest.warns(UserWarning, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_negative_out_strength(self):
        """ Test that an error is raised if out_strength contains negative
        values in either id, label or value.
        """
        tmp = out_strength.copy().astype([('label', np.int8),
                                          ('id', np.int8),
                                          ('value', np.float64)])

        tmp.label[1] = -1
        msg = "Out strength labels must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        tmp.label[1] = out_strength.label[1]
        tmp.id[2] = -tmp.id[2]
        msg = "Out strength ids must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

        tmp.id[2] = -tmp.id[2]
        tmp.value = -tmp.value
        msg = "Out strength values must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=tmp,
                                  in_strength=in_strength,
                                  num_edges=num_edges)

    def test_negative_in_strength(self):
        """ Test that an error is raised if in_strength contains negative
        values in either id, label or value.
        """
        tmp = in_strength.copy().astype([('label', np.int8),
                                         ('id', np.int8),
                                         ('value', np.float64)])

        tmp.label[2] = -tmp.label[2]
        msg = "In strength labels must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=tmp,
                                  num_edges=num_edges)

        tmp.label[2] = -tmp.label[2]
        tmp.id[2] = -tmp.id[2]
        msg = "In strength ids must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=tmp,
                                  num_edges=num_edges)

        tmp.id[2] = -tmp.id[2]
        tmp.value = -tmp.value
        msg = "In strength values must contain positive values only."
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=tmp,
                                  num_edges=num_edges)

    def test_wrong_num_edges(self):
        """ Check that wrong initialization of num_edges results in an error.
        """
        msg = ('Number of edges must be a number or a numpy array of length'
               ' equal to the number of labels.')
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges='3')
        with pytest.raises(Exception, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=np.array([1, 2]))

        msg = 'Number of edges must contain only positive values.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=-324)
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  num_edges=-num_edges_label)

        msg = 'Either num_edges or z must be set.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength)

    def test_wrong_z(self):
        """ Check that the passed z adheres to format.
        """
        msg = ('z must be a number or an array of length equal to the number '
               'of labels.')
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  z='three')
        with pytest.raises(AssertionError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  z=np.array([0, 1]))

        msg = 'z must contain only positive values.'
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  z=-1)
        with pytest.raises(ValueError, match=msg):
            ge.StripeFitnessModel(num_vertices=num_vertices,
                                  num_labels=num_labels,
                                  out_strength=out_strength,
                                  in_strength=in_strength,
                                  z=-z_label)


class TestStripeFitnessModelFit():
    # def test_solver_newton_single_z(self):
    #     """ Check that the newton solver is fitting the z parameters
    #     correctly. """
    #     model = ge.StripeFitnessModel(g, per_label=False)
    #     model.fit(method="newton")
    #     exp_num_edges = model.expected_num_edges()
    #     np.testing.assert_allclose(num_edges_label, exp_num_edges,
    #                                atol=1e-5, rtol=0)
    #     np.testing.assert_allclose(z, model.z, atol=0, rtol=1e-6)

    def test_solver_newton_multi_z(self):
        """ Check that the newton solver is fitting the z parameters
        correctly. """
        model = ge.StripeFitnessModel(g)
        model.fit(method="newton")
        exp_num_edges_label = model.expected_num_edges_label()
        np.testing.assert_allclose(num_edges_label, exp_num_edges_label,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_label, model.z, atol=0, rtol=1e-5)

    # def test_solver_invariant_single_z(self):
    #     """ Check that the newton solver is fitting the z parameters
    #     correctly for the invariant case. """
    #     model = ge.StripeFitnessModel(g, per_label=False, scale_invariant=True)
    #     model.fit(method="newton")
    #     exp_num_edges = model.expected_num_edges()
    #     np.testing.assert_allclose(num_edges, exp_num_edges,
    #                                atol=1e-5, rtol=0)
    #     np.testing.assert_allclose(z_inv, model.z, atol=0, rtol=1e-6)

    def test_solver_invariant_multi_z(self):
        """ Check that the newton solver is fitting the z parameters
        correctly for the invariant case. """
        model = ge.StripeFitnessModel(g, scale_invariant=True)
        model.fit(method="newton")
        exp_num_edges_label = model.expected_num_edges_label()
        np.testing.assert_allclose(num_edges_label, exp_num_edges_label,
                                   atol=1e-5, rtol=0)
        np.testing.assert_allclose(z_inv_lbl, model.z, atol=0, rtol=1e-6)

    # def test_solver_fixed_point_single_z(self):
    #     """ Check that the fixed-point solver is fitting the z parameters
    #     correctly.
    #     """
    #     model = ge.StripeFitnessModel(g, per_label=False)
    #     model.fit(method="fixed-point", max_iter=100, xtol=1e-5)
    #     exp_num_edges = model.expected_num_edges()
    #     np.testing.assert_allclose(num_edges, exp_num_edges,
    #                                atol=1e-4, rtol=0)
    #     np.testing.assert_allclose(z, model.z, atol=0, rtol=1e-4)

    def test_solver_fixed_point_multi_z(self):
        """ Check that the fixed-point solver is fitting the z parameters
        correctly.
        """
        model = ge.StripeFitnessModel(g)
        model.fit(method="fixed-point", max_iter=50000, xtol=1e-4)
        exp_num_edges_label = model.expected_num_edges_label()
        np.testing.assert_allclose(num_edges_label, exp_num_edges_label,
                                   atol=1e-3, rtol=0)

    # def test_solver_min_degree_single_z(self):
    #     """ Check that the min_degree solver converges.
    #     """
    #     model = ge.FitnessModel(g, per_label= False, min_degree=True)
    #     model.fit(tol=1e-6, max_iter=500)
    #     exp_num_edges = model.expected_num_edges()
    #     np.testing.assert_allclose(num_edges, exp_num_edges,
    #                                atol=1e-5, rtol=0)
    #     assert np.all(model.expected_out_degree() >= 1 - 1e-5)
    #     assert np.all(model.expected_in_degree() >= 1 - 1e-5)
    #     np.testing.assert_allclose(z, model.z, atol=0, rtol=1e-6)
    #     np.testing.assert_allclose(1.0, model.alpha, atol=0, rtol=1e-6)

    # def test_solver_min_degree_multi_z(self):
    #     """ Check that the min_degree solver converges.
    #     """
    #     model = ge.FitnessModel(g, min_degree=True)
    #     model.fit(tol=1e-6, max_iter=500)
    #     exp_num_edges = model.expected_num_edges()
    #     np.testing.assert_allclose(num_edges, exp_num_edges,
    #                                atol=1e-5, rtol=0)
    #     assert np.all(model.expected_out_degree() >= 1 - 1e-5)
    #     assert np.all(model.expected_in_degree() >= 1 - 1e-5)
    #     np.testing.assert_allclose(z_label, model.z, atol=0, rtol=1e-6)
    #     np.testing.assert_allclose(np.ones(num_labels),
    #                                model.alpha, atol=0, rtol=1e-6)

    def test_solver_with_init(self):
        """ Check that it works with a given initial condition.
        """
        # model = ge.StripeFitnessModel(g, per_label=False)
        # model.fit(z0=1e-14)
        # exp_num_edges = model.expected_num_edges()
        # np.testing.assert_allclose(num_edges, exp_num_edges,
        #                            atol=1e-5, rtol=0)
        # np.testing.assert_allclose(z, model.z, atol=0, rtol=1e-6)

        model = ge.StripeFitnessModel(g, per_label=True)
        model.fit(z0=1e-14*np.ones(num_labels))
        exp_num_edges_label = model.expected_num_edges_label()
        np.testing.assert_allclose(num_edges_label, exp_num_edges_label,
                                   atol=1e-5, rtol=0)

    # def test_solver_with_wrong_init(self):
    #     """ Check that it raises an error with a negative initial condition.
    #     """
    #     msg = "z0 must contain only positive values."
    #     with pytest.raises(ValueError, match=msg):
    #         model = ge.StripeFitnessModel(g, per_label=False)
    #         model.fit(z0=-1)
    #     with pytest.raises(ValueError, match=msg):
    #         model = ge.StripeFitnessModel(g, per_label=True)
    #         model.fit(z0=-np.ones(num_labels))

    #     msg = 'z0 must be a number.'
    #     with pytest.raises(ValueError, match=msg):
    #         model = ge.StripeFitnessModel(g, per_label=False)
    #         model.fit(z0=np.array([0, 1]))

    #     msg = 'z0 must be an array with length equal to num_labels.'
    #     with pytest.raises(ValueError, match=msg):
    #         model = ge.StripeFitnessModel(g, per_label=True)
    #         model.fit(z0=np.array([0, 1]))

    def test_wrong_method(self):
        """ Check that wrong methods names return an error.
        """
        model = ge.StripeFitnessModel(g)
        msg = "The selected method is not valid."
        with pytest.raises(ValueError, match=msg):
            model.fit(method="wrong")

    def test_method_incompatibility(self):
        """ Check that an error is raised when trying to use the wrong method.
        """
        model = ge.StripeFitnessModel(g, scale_invariant=True)
        msg = ('Fixed point solver not supported for scale '
               'invariant functional.')
        with pytest.raises(Exception, match=msg):
            model.fit(method="fixed-point", max_iter=100, xtol=1e-5)

        # model = ge.StripeFitnessModel(g, min_degree=True)
        # msg = ('Method not recognised for solver with min degree '
        #        'constraint, using default SLSQP.')
        # with pytest.warns(UserWarning, match=msg):
        #     model.fit(method="newton")

        msg = 'Cannot constrain min degree in scale invariant model.'
        with pytest.raises(Exception, match=msg):
            model = ge.StripeFitnessModel(
              g, scale_invariant=True, min_degree=True)


class TestFitnessModelMeasures():
    def test_exp_n_edges_single_z(self):
        """ Check expected edges is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z)
        n_e = model.expected_num_edges()
        np.testing.assert_allclose(n_e,
                                   num_edges,
                                   rtol=1e-5)

    def test_exp_n_edges_multi_z(self):
        """ Check expected edges is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)
        n_e = model.expected_num_edges()
        np.testing.assert_allclose(n_e,
                                   5.153923,
                                   rtol=1e-5)

    def test_exp_n_edges_label_single_z(self):
        """ Check expected edges is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z)
        n_e = model.expected_num_edges_label()
        np.testing.assert_allclose(n_e,
                                   num_edges_label,
                                   rtol=1e-5)

    def test_exp_n_edges_label_multi_z(self):
        """ Check expected edges is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)
        n_e = model.expected_num_edges_label()
        np.testing.assert_allclose(n_e,
                                   num_edges_label,
                                   rtol=1e-5)

    # def test_exp_out_degree_single_z(self):
    #     """ Check expected d_out is correct. """
    #     model = ge.StripeFitnessModel(num_vertices=num_vertices,
    #                                   num_labels=num_labels,
    #                                   out_strength=out_strength,
    #                                   in_strength=in_strength,
    #                                   z=z)
    #     d_out = model.expected_out_degree()
    #     np.testing.assert_allclose(
    #         d_out, np.array([1.0, 1.0, 1.0, 1.0]),
    #         rtol=1e-5)

    def test_exp_out_degree_multi_z(self):
        """ Check expected d_out is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)
        d_out = model.expected_out_degree()
        np.testing.assert_allclose(
            d_out, np.array([1.947547, 1.154435, 0.999992, 1.051948]),
            rtol=1e-5)

    # def test_exp_in_degree_single_z(self):
    #     """ Check expected d_in is correct. """
    #     model = ge.StripeFitnessModel(num_vertices=num_vertices,
    #                                   num_labels=num_labels,
    #                                   out_strength=out_strength,
    #                                   in_strength=in_strength,
    #                                   z=z)
    #     d_in = model.expected_in_degree()
    #     np.testing.assert_allclose(
    #         d_in, np.array([1.0, 1.0, 1.0, 0.0]), rtol=1e-5)

    def test_exp_in_degree_multi_z(self):
        """ Check expected d_in is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)
        d_in = model.expected_in_degree()
        np.testing.assert_allclose(
            d_in, np.array([0.999992, 2.999446, 1.154485, 0.0]), rtol=1e-5)

    # def test_exp_out_degree_by_label_single_z(self):
    #     """ Check expected d_out is correct. """
    #     model = ge.StripeFitnessModel(num_vertices=num_vertices,
    #                                   num_labels=num_labels,
    #                                   out_strength=out_strength,
    #                                   in_strength=in_strength,
    #                                   z=z)
    #     d_out = model.expected_out_degree_by_label()
    #     np.testing.assert_allclose(
    #         d_out, np.array([1.0, 1.0, 1.0, 1.0]),
    #         rtol=1e-5)

    def test_exp_out_degree_by_label_multi_z(self):
        """ Check expected d_out is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)
        d_ref = np.array([(0, 0, 1.947547),
                          (0, 1, 0.154443),
                          (0, 3, 0.898008),
                          (1, 2, 0.999992),
                          (2, 3, 0.999992),
                          (3, 1, 0.999992)],
                         dtype=[('label', 'u1'),
                                ('id', 'u1'),
                                ('value', '<f8')]).view(type=np.recarray)
        d_out = model.expected_out_degree_by_label()
        np.testing.assert_allclose(d_out.label, d_ref.label, rtol=0)
        np.testing.assert_allclose(d_out.id, d_ref.id, rtol=0)
        np.testing.assert_allclose(d_out.value, d_ref.value, rtol=1e-5)

    # def test_exp_in_degree_by_label_single_z(self):
    #     """ Check expected d_in is correct. """
    #     model = ge.StripeFitnessModel(num_vertices=num_vertices,
    #                                   num_labels=num_labels,
    #                                   out_strength=out_strength,
    #                                   in_strength=in_strength,
    #                                   z=z)
    #     d_in = model.expected_in_degree_by_label()
    #     np.testing.assert_allclose(
    #         d_in, np.array([1.0, 1.0, 1.0, 0.0]), rtol=1e-5)

    def test_exp_in_degree_by_label_multi_z(self):
        """ Check expected d_in is correct. """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)
        d_ref = np.array([(0, 1, 1.845514),
                          (0, 2, 1.154485),
                          (1, 1, 0.999992),
                          (2, 1, 0.999992),
                          (3, 0, 0.999992)],
                         dtype=[('label', 'u1'),
                                ('id', 'u1'),
                                ('value', '<f8')]).view(type=np.recarray)
        d_in = model.expected_in_degree_by_label()
        np.testing.assert_allclose(d_in.label, d_ref.label, rtol=0)
        np.testing.assert_allclose(d_in.id, d_ref.id, rtol=0)
        np.testing.assert_allclose(d_in.value, d_ref.value, rtol=1e-5)


class TestFitnessModelSample():
    # def test_sampling_single_z(self):
    #     """ Check that properties of the sample correspond to ensemble.
    #     """
    #     model = ge.StripeFitnessModel(num_vertices=num_vertices,
    #                                   num_labels=num_labels,
    #                                   out_strength=out_strength,
    #                                   in_strength=in_strength,
    #                                   z=z)

    #     samples = 10000
    #     s_n_e = np.empty((samples, num_labels))
    #     for i in range(samples):
    #         sample = model.sample()
    #         s_n_e[i] = sample.num_edges_label
    #         assert np.all(sample.num_labels == num_labels)
    #         assert np.all(sample.num_vertices == num_vertices)

    #     s_n_e = np.average(s_n_e, axis=0)
    #     np.testing.assert_allclose(s_n_e, num_edges, atol=1e-1, rtol=0)

    def test_sampling_multi_z(self):
        """ Check that properties of the sample correspond to ensemble.
        """
        model = ge.StripeFitnessModel(num_vertices=num_vertices,
                                      num_labels=num_labels,
                                      out_strength=out_strength,
                                      in_strength=in_strength,
                                      z=z_label)

        samples = 10000
        s_n_e = np.empty((samples, num_labels))
        for i in range(samples):
            sample = model.sample()
            s_n_e[i] = sample.num_edges_label
            assert np.all(sample.num_labels == num_labels)
            assert np.all(sample.num_vertices == num_vertices)

        s_n_e = np.average(s_n_e, axis=0)
        np.testing.assert_allclose(s_n_e, num_edges_label, atol=1e-1, rtol=0)
