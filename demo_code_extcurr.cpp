#include "core/configuration.hpp"
#include "core/engine.hpp"
#include "ExternalCurrent.hpp"

#include <boost/numeric/odeint.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace insilico;
using namespace std;

ExternalCurrent ec;

class I_Na {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gna = 120, ena = 55;

    int v_index = engine::neuron_index(index, "v");
    int m_index = engine::neuron_index(index, "m");
    int h_index = engine::neuron_index(index, "h");
    
    double v = variables[v_index];
    double m = variables[m_index];
    double h = variables[h_index];
    
    double alpha_m = (2.5 - 0.1 * v) / (exp(2.5-0.1 * v) - 1.0);
    double beta_m  = 4.0 * exp(-v / 18.0);
    double alpha_h = 0.07 * exp(-v / 20.0);
    double beta_h  = 1.0 / (exp(3 - 0.1 * v) + 1);

    dxdt[m_index] = (alpha_m * (1-m) - beta_m * m);
    dxdt[h_index] = (alpha_h * (1-h) - beta_h * h);

    engine::current_value(index, "I_Na", (gna * pow(m, 3) * h * (v - ena)));
  }
};

class I_K {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gk = 36, ek = -75;

    int v_index = engine::neuron_index(index, "v");
    int n_index = engine::neuron_index(index, "n");

    double v = variables[v_index];
    double n = variables[n_index];

    double alpha_n = (0.1 - 0.01 * v) / (exp(1 - 0.1 * v) - 1.0);
    double beta_n  = 0.125 * exp(-v / 80.0);

    dxdt[n_index]=(alpha_n*(1 - n)-beta_n * n);

    engine::current_value(index, "I_K", (gk * pow(n,4) * (v - ek)));
  }
};

class I_Leak {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gl = 0.3, el = -66;

    int v_index = engine::neuron_index(index, "v");
    double v = variables[v_index];

    engine::current_value(index, "I_Leak", (gl * (v - el)));
  }
};

class HH_Neuron {
 public:
  static void ode_set(state_type &variables, state_type &dxdt, const double t, int index) {
    int v_index = engine::neuron_index(index, "v");

    I_Na::current(variables, dxdt, t, index);
    I_K::current(variables, dxdt, t, index);
    I_Leak::current(variables, dxdt, t, index);

    double I_Na = engine::current_value(index, "I_Na");
    double I_K = engine::current_value(index, "I_K");
    double I_Leak = engine::current_value(index, "I_Leak");
    double I_Ext = ec.I_Ext(t);
    
    dxdt[v_index] = - I_Na - I_K - I_Leak + I_Ext;
  }
};

void engine::operator()(state_type &variables, state_type &dxdt, const double time) {
  HH_Neuron::ode_set(variables, dxdt, time, 0);
}

void configuration::observer::operator()(state_type &variables, const double t) {
  vector<int> v_indices = engine::get_indices("v");
  vector<int> m_indices = engine::get_indices("m");
  vector<int> h_indices = engine::get_indices("h");
  vector<int> n_indices = engine::get_indices("n");
  
  observer::outfile << t;
  
  for(vector<double>::size_type index = 0; index < v_indices.size() ; ++index) {
    observer::outfile << ',' << variables[v_indices[index]];
    observer::outfile << ',' << variables[m_indices[index]];
    observer::outfile << ',' << variables[h_indices[index]];
    observer::outfile << ',' << variables[n_indices[index]];
  }
  
  observer::outfile << '\n';
}

int main(int argc, char **argv) {
  configuration::initialize(argc, argv);
  state_type variables = engine::get_variables();

  ec.set_ext_currect(0, 100, 0);
  //ec.set_ext_currect(30, 40, 1);
  //ec.set_ext_currect(70, 80, 5);

  using namespace boost::numeric::odeint;
  integrate_const(runge_kutta4<state_type>(),
                  engine(), variables,
                  0.0, 100.0, 0.006,
                  configuration::observer(configuration::outstream));

  configuration::finalize();
  return 0;
}
