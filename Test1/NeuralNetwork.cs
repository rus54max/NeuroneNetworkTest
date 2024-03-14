using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;
using System.Text;
using System.Threading.Tasks;

namespace Test1
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayer();
            CreateOutputLayer();
        }

        public Neuron FeedForward(List<double> inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfrerInput();
            if (Topology.OutputCount == 0)
                return Layers.Last().Neurons[0];
            else
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();

        }

        public void FeedForwardAllLayersAfrerInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                Layer layer = Layers[i];
                List<double> prevSignals = Layers[i - 1].GetSignals(); //список сигналов пердыдущего слоя
                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.FeedForward(prevSignals); //считаем в текущем слое все нейроны
                }
            }

        }

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (int x = 0; x < inputSignals.Count; x++)
            {
                List<double> signal = new List<double>() { inputSignals[x] };
                Neuron neuron = Layers[0].Neurons[x];
                neuron.FeedForward(signal);
            }
        }

        private void CreateOutputLayer()
        {
            List<Neuron> outputNeurons = new List<Neuron>();
            Layer lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                Neuron neuron = new Neuron(lastLayer.Count, EnNeuronType.Output);
                outputNeurons.Add(neuron);
            }
            Layer layer = new Layer(outputNeurons, EnNeuronType.Output);
            Layers.Add(layer);
        }

        private void CreateHiddenLayer()
        {
            for (int counterLayers = 0; counterLayers < Topology.HiddenLayers.Count; counterLayers++)
            {
                List<Neuron> neurons = new List<Neuron>();
                Layer lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[counterLayers]; i++)
                {
                    Neuron neuron = new Neuron(lastLayer.Count);
                    neurons.Add(neuron);
                }
                Layer layer = new Layer(neurons, EnNeuronType.Output);
                Layers.Add(layer);

            }
        }

        private void CreateInputLayer()
        {
            List<Neuron> inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                Neuron neuron = new Neuron(1, EnNeuronType.Input);
                inputNeurons.Add(neuron);
            }
            Layer layer = new Layer(inputNeurons, EnNeuronType.Input);
            Layers.Add(layer);
        }
    }
}
