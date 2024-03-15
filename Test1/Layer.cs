using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test1
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public EnNeuronType NeuronType { get; }
        public int Count => Neurons?.Count ?? 0;

        public Layer(List<Neuron> neurons, EnNeuronType typeNeuron = EnNeuronType.Normal)
        {
            Neurons = neurons;
            NeuronType = typeNeuron;
        }

        public List<double> GetSignals()
        {
            List<double> signals = new List<double>();
            foreach(Neuron neuron in Neurons)
            {
                signals.Add(neuron.Output);
            }
            return signals;
        }
        public override string ToString()
        {
            return NeuronType.ToString();
        }
    }


}
