/*
 main.cpp - insilico's example using neuron and synapse for illustrations

 Copyright (C) 2014-2015 Pranav Kulkarni, Collins Assisi Lab, IISER, Pune <pranavcode@gmail.com>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "parallel/mpi.hpp"

#include "core/configuration.hpp"
#include "core/engine.hpp"

#include "neuron/N_SquidAxon_HH1952.hpp"
#include "synapse/S_DefaultSynapse.hpp"

#include <boost/numeric/odeint.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace insilico;
using namespace std;

void engine::operator()(state_type &variables, state_type &dxdt, const double time) {
  int neuron_count = engine::neuron_count();

  // work distribution among mpi processes
  int my_neuron_start = neuron_count / mpi::size * mpi::rank;
  int my_neuron_end = neuron_count / mpi::size * (mpi::rank + 1);

  if(mpi::rank == (mpi::size - 1) && my_neuron_end < neuron_count ) {
    my_neuron_end = neuron_count;
  }

  int synapse_count = engine::synapse_count();

  int my_synapse_start = synapse_count / mpi::size * mpi::rank;
  int my_synapse_end = synapse_count / mpi::size * (mpi::rank + 1);

  if(mpi::rank == (mpi::size - 1) && my_synapse_end < synapse_count ) {
    my_synapse_end = synapse_count;
  }

  for(int neuron_index = my_neuron_start; neuron_index < my_neuron_end; ++neuron_index) {
    N_SquidAxon_HH1952::ode_set(variables, dxdt, time, neuron_index);
  }
  for(int synapse_index = my_synapse_start; synapse_index < my_synapse_end; ++synapse_index) {
    S_DefaultSynapse::ode_set(variables, dxdt, time, synapse_index);
  }
}

void configuration::observer::operator()(state_type &variables, const double t) {
  mpi::engine::synchronize(variables);

  if(mpi::rank == MASTER) {
    vector<int> indices = engine::get_indices("v");
    observer::outfile << setprecision(2) << fixed << t;
    for(int index : indices) {
      observer::outfile << ',' << setprecision(6) << fixed << variables[index];
    }
    observer::outfile<<endl;
  }
};

int main(int argc, char **argv) {
  configuration::initialize(argc, argv);
  state_type variables = engine::get_variables();

  using namespace boost::numeric::odeint;
  integrate_const(runge_kutta4<state_type>(), engine(), variables,
                  0.0, 100.0, 0.05, configuration::observer(configuration::outstream));

  configuration::finalize();  
  return 0;
}
