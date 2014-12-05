/*
 main.cpp - nsim source supporting models and main()

 Copyright (C) 2014 Collins Assisi, Collins Assisi Lab, IISER, Pune
 Copyright (C) 2014 Pranav Kulkarni, Collins Assisi Lab, IISER, Pune <pranavcode@gmail.com>
 Copyright (C) 2014 Arun Neru, Collins Assisi Lab, IISER, Pune <areinsdel@gmail.com>

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

#include <boost/numeric/odeint.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nsim/network/nnet.hpp>

using namespace boost;
using namespace std;

typedef vector<long double> state_type;

struct configuration {
  ofstream &stream;
  configuration(ofstream &file): stream(file) {}
  void operator()(const state_type &variables, const double t) {
    vector<long> indices = nnet::get_indices("v");
    assert(stream.is_open());
    stream<<t;
    for(vector<long>::size_type iter = 0; iter < indices.size(); ++iter) {
      stream<<','<<variables[indices[iter]];
    }
    stream<<endl;
  }
};

class na_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) { 
    long v_index = nnet::neuron_index(index, "v");
    long m_index = nnet::neuron_index(index, "m");
    long h_index = nnet::neuron_index(index, "h");
    
    double V = variables[v_index];

    double m = variables[m_index];
    double h = variables[h_index];
    
    double alpha_m = -0.1*(V+23.0)/(exp(-0.1*(V+23.0))-1.0);
    double beta_m  = 4.0*exp(-(V+48)/18.0);
    double alpha_h = 0.07*exp(-(V+37.0)/20.0);
    double beta_h  = 1.0/(exp(-0.1*(V+7.0))+1.0);
	
    dxdt[m_index]= alpha_m*(1-m)-beta_m*m;
    dxdt[h_index]= alpha_h*(1-h)-beta_h*h;
  }
};

class na_fast_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) { 
    long v_index = nnet::neuron_index(index, "v");
    long h_index = nnet::neuron_index(index, "h");
    
    double V = variables[v_index];
    double h = variables[h_index];	
    double phi = 5.0;
 
    double alpha_h = 0.07*exp(-(V+58)/20.0);
    double beta_h  = 1.0/(exp(-0.1*(V+28.0))+1.0);

    dxdt[h_index]= phi*(alpha_h*(1-h)-beta_h*h);
  }
};

class na_p_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) { 
    long v_index = nnet::neuron_index(index, "v");
    long ms_index = nnet::neuron_index(index, "ms");

    double V = variables[v_index];
    double ms = variables[ms_index];
    
    double alpha_ms = 1.0/(0.15*(1.0+exp(-(V+38.0)/6.5)));
    double beta_ms  = exp(-(V+38.0)/6.5)/(0.15*(1.0+exp(-(V+38.0)/6.5)));
	
    dxdt[ms_index]= alpha_ms*(1-ms)-beta_ms*ms;
  }
}; 

class k_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) {
    long v_index = nnet::neuron_index(index, "v");
    long n_index = nnet::neuron_index(index, "n");

    double V = variables[v_index];
    double n = variables[n_index];

    double alpha_n = -0.01*(V+27.0)/(exp(-0.1*(V+27.0))-1.0);
    double beta_n = 0.125*exp(-(V+37.0)/80.0);

    dxdt[n_index]=(alpha_n*(1 - n)-beta_n * n);
  }
};

class k_fast_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) {
    long v_index = nnet::neuron_index(index, "v");
    long n_index = nnet::neuron_index(index, "n");

    double V = variables[v_index];
    double n = variables[n_index];

    double alpha_n = 0.01*(V+34.0)/(1-exp(-0.1*(V+34)));
    double beta_n = 0.125*exp(-(V+44.0)/80.0);

    dxdt[n_index]=(alpha_n*(1 - n)-beta_n * n);
  }
};

class h_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) { 
    long v_index = nnet::neuron_index(index, "v");
    long mhf_index = nnet::neuron_index(index, "mhf");
    long mhs_index = nnet::neuron_index(index, "mhs");
    
    double V = variables[v_index];

    double mhf = variables[mhf_index];
    double mhs = variables[mhs_index];
    
    double mhfinf = 1.0/(1.0+exp((V+79.2)/9.78));
    double mhftau = 0.51/(exp((V-1.7)/10.0)+exp(-(V+340.0)/52.0))+1.0;

    double mhsinf = 1.0/(1.0+exp((V+71.3)/7.9));
    double mhstau = 5.6/(exp((V-1.7)/14.0)+exp(-(V+260.0)/43.0))+1.0;
	
    dxdt[mhf_index]= (mhfinf-mhf)/mhftau;
    dxdt[mhs_index]= (mhsinf-mhs)/mhstau;
  }
};

class ca_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) { 
    long v_index = nnet::neuron_index(index, "v");
    long mtc_index = nnet::neuron_index(index, "mtc");
    long htc_index = nnet::neuron_index(index, "htc");
    
    double V = variables[v_index];

    double mtc = variables[mtc_index];
    double htc = variables[htc_index];

    double mtc_inf = 1.0/(1.0+exp(-(V+20.0)/6.5));
    double mtc_tau = 1+(V+30)*0.014;
		
    double htc_inf = 1.0/(1.0+exp((V+25.0)/12.0));
    double htc_tau = 0.3*exp((V-40.0)/13.0)+0.002*exp(-(V-60.0)/29.0);
	
    dxdt[mtc_index]= (mtc_inf-mtc)/mtc_tau;
    dxdt[htc_index]= (htc_inf-htc)/htc_tau;
  }
};

class cak_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) { 
    long mck_index = nnet::neuron_index(index, "mck");
    long ca_index = nnet::neuron_index(index, "ca");
    
    double mck = variables[mck_index];
    double Ca = variables[ca_index];
	
    double tad = pow(2.3,(25.0-23.0)/10.0);
    double ra = 0.01;
    double rb = 0.02;
	
    double a = ra*Ca;
    double b = rb;

    double mck_tau =1.0/(tad*(a+b));
    double mck_inf = a/(a+b);
    dxdt[mck_index]= (mck_inf-mck)/mck_tau;
  }
};

class ca_in_conc {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) { 
    long v_index = nnet::neuron_index(index, "v");
    long mtc_index = nnet::neuron_index(index, "mtc");
    long htc_index = nnet::neuron_index(index, "htc");
    long ca_index = nnet::neuron_index(index, "ca");
    
    //long gca_index = nnet::get_index(index, "gca", NEURON);
    //long eca_index = nnet::get_index(index, "eca", NEURON);
	
    double V = variables[v_index];
    double mtc = variables[mtc_index];
    double htc = variables[htc_index];
    double Ca = variables[ca_index];
    double gca = 2.0;
    double eca = 140.0;
    double drive0 =15.0/(2.0*96489.0);
    double ca_q = 2.4*pow(10.0,-4.0);
	
    double tau_r = 231*5.0;
    double d = 0.1;
    double iT = gca*pow(mtc,2.0)*htc*(V-eca);
    double drive = -drive0*iT/d;

    if(drive<0.0) {
      drive=0.0;
    }
    
    dxdt[ca_index]= drive+(ca_q-Ca)/tau_r;
  }
};

class leak_conductance {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t, long index) {}
};

class hodgkin_huxley_neuron {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t,
               long index) {

    long v_index = nnet::neuron_index(index, "v");
    long n_index = nnet::neuron_index(index, "n");
    long m_index = nnet::neuron_index(index, "m");
    long h_index = nnet::neuron_index(index, "h");
    long ms_index = nnet::neuron_index(index, "ms");
    long mhf_index = nnet::neuron_index(index, "mhf");
    long mhs_index = nnet::neuron_index(index, "mhs");
    long mtc_index = nnet::neuron_index(index, "mtc");
    long htc_index = nnet::neuron_index(index, "htc");
    long mck_index = nnet::neuron_index(index, "mck");
	
        
    double V = variables[v_index];
    double n = variables[n_index];
    double m = variables[m_index];

    double h = variables[h_index];	
    double ms = variables[ms_index];
    double mhf = variables[mhf_index];
    double mhs = variables[mhs_index];
    double mtc = variables[mtc_index];
    double htc = variables[htc_index];
    double mck = variables[mck_index];

    double gna = nnet::neuron_value(index, "gna");
    double ena = nnet::neuron_value(index, "ena");
    double gk = nnet::neuron_value(index, "gk");
    double ek = nnet::neuron_value(index, "ek");
    double gl = nnet::neuron_value(index, "gl");
    double el = nnet::neuron_value(index, "el");

    double iext = nnet::neuron_value(index, "iext");
    double gnap = 0.5;
    double gh = 1.5;
    double eh = -20.0;
    double gca = 2.0;
    double eca = 140.0;
    double tad = pow(2.3,(25.0-23.0)/10.0);
    double gkca = 0.3;
    double ekca = -90.0;
  
    vector<long> g1_indices = nnet::get_pre_neuron_indices(index, "g1", SYNAPSE);
    vector<long> esyn_indices = nnet::get_pre_neuron_indices(index, "esyn", SYNAPSE);
    double isyn = 0;

    for(vector<long>::size_type iterator = 0; iterator < g1_indices.size(); ++iterator) {
      isyn = isyn + variables[g1_indices.at(iterator)] * (V - variables[esyn_indices.at(iterator)]); 
    }

    // ODE
    dxdt[v_index] = -(gna*pow(m,3)*h+gnap*ms)*(V-ena)-gk*pow(n,4)*(V-ek)-gh*(0.65*mhf+0.35*mhs)*(V-eh)-0.0*gca*pow(mtc,2.0)*htc*(V-eca)-0.0*0.15*tad*gkca*mck*(V-ekca)-gl*(V-el)+iext-isyn;

    na_conductance::ode_set(variables, dxdt, t, index);
    k_conductance::ode_set(variables, dxdt, t, index);
    na_p_conductance::ode_set(variables, dxdt, t, index);
    h_conductance::ode_set(variables, dxdt, t, index);
    ca_conductance::ode_set(variables, dxdt, t, index);
    cak_conductance::ode_set(variables, dxdt, t, index);
    ca_in_conc::ode_set(variables, dxdt, t, index);
    leak_conductance::ode_set(variables, dxdt, t, index);
  }
};

class hodgkin_huxley_inhibitory_neuron {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t,
                      long index) {
    long v_index = nnet::neuron_index(index, "v");
    long n_index = nnet::neuron_index(index, "n");
    //long m_index = nnet::neuron_index(index, "m");
    long h_index = nnet::neuron_index(index, "h");
        
    double V = variables[v_index];
    double n = variables[n_index];
    //double m = variables[m_index];
    double h = variables[h_index];	
	
	
    double gna = 52.0;
    double ena = 55.0;
    double gk = 11.0;
    double ek = -90.0;
    double i_gl = 0.1; 
    double el = -65.0;

    double iext = nnet::neuron_value(index, "iext");
	
    vector<long> g1_indices = nnet::get_pre_neuron_indices(index, "g1", SYNAPSE);
    vector<long> esyn_indices = nnet::get_pre_neuron_indices(index, "esyn", SYNAPSE);
    double isyn = 0;

    for(vector<long>::size_type iterator = 0; iterator < g1_indices.size(); ++iterator) {
      isyn = isyn + variables[g1_indices.at(iterator)] * (V - variables[esyn_indices.at(iterator)]); 
    }

    //cout<<"V is ="<<V<<"\n";
    double alpha = 0.1*(V+35.0)/(1-exp(-(V+35.0)/10.0));
    double beta = 4.0*exp(-(V+60.0)/18.0);
    double m = alpha/(alpha+beta);
    //cout<<"m is "<<m<<"\t";
	 
    // ODE
    //cout<<"isyn is"<<isyn<<"\n";
    dxdt[v_index] = -gna*pow(m,3)*h*(V-ena)-gk*pow(n,4)*(V-ek)-i_gl*(V-el)+iext-isyn;

    na_fast_conductance::ode_set(variables, dxdt, t, index);
    k_fast_conductance::ode_set(variables, dxdt, t, index);
    //leak_conductance::ode_set(variables, dxdt, t, index);
  }
};

class synapse_x {
 public:
  static void ode_set(const state_type &variables, state_type &dxdt, const double t,
                      long index) {
    long g1_index = nnet::synapse_index(index, "g1");
    long g2_index = nnet::synapse_index(index, "g2");
    long pre_index = nnet::synapse_index(index, "pre");
    long neuron_index = variables[pre_index];

    long last_spiked_index = nnet::synapse_index(index, "last_spike");
    long v_index = nnet::neuron_index(neuron_index, "v");

    double g1 = variables[g1_index];
    double g2 = variables[g2_index];

    double last_spiked = variables[last_spiked_index];
    double V = variables[v_index];
    double def_delay = .004;
    double thresh = 20.0;
    double xt = 0.0;
	
    if((V > thresh) && (t-last_spiked)>def_delay){
      xt = 1.0;
      dxdt[last_spiked_index] = t*(1.0/0.05);
    }

    // constants from file
    double tau1 = nnet::synapse_value(index, "tau1");
    double tau2 = nnet::synapse_value(index, "tau2");
    double gsyn = nnet::synapse_value(index, "gsyn");

    dxdt[g1_index] = g2;
    dxdt[g2_index] = -((tau1+tau2)/(tau1*tau2))*g2-g1+gsyn*xt;
  }
};

void nnet::operator()(const state_type &variables, state_type &dxdt,
                      const double time) {
  long network_size = nnet::neuron_count();
  long synapse_count = nnet::synapse_count();

  for(long neuron_index = 0; neuron_index < network_size-1; ++neuron_index) {
    hodgkin_huxley_neuron::ode_set(variables, dxdt, time, neuron_index);
  }
  hodgkin_huxley_inhibitory_neuron::ode_set(variables, dxdt, time, network_size-1);
  // seperate synapse division

  for(long synapse_index = 0; synapse_index < synapse_count; ++synapse_index) {
    synapse_x::ode_set(variables, dxdt, time, synapse_index);
  }
}

int main(int argc, char* argv[]) {
  if(argc != 2) {
    cout<<"Usage: "<<argv[0]<<" <outputfile>.dat"<<endl;
    return -1;
  }
  
  nnet network;
  nnet::read("nsets.conf","ssets.conf");
  state_type variables = nnet::get_variables();
  string output_filename = argv[1];
  ofstream output_file(output_filename);
  assert(output_file.is_open());
  
  using namespace boost::numeric::odeint;
  integrate_const(runge_kutta_dopri5<state_type>(), network, variables,
                  0.0, 2000.0, 0.01, configuration(output_file));

  output_file.close();
  return 0;
}
