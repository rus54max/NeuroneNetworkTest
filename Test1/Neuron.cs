using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test1
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public EnNeuronType NeuronType { get; }
        public double Output { get; private set; }

        public Neuron(int inputCount, EnNeuronType neuronType = EnNeuronType.Normal)
        {
            NeuronType = neuronType;
            Weights = new List<double>();
            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(1);

            }
        }

        public double FeedForward(List<double> inputs)
        {
            double sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
                sum += inputs[i] * Weights[i];
            if (NeuronType != EnNeuronType.Input)
                Output = Sigmoid(sum);
            else
                Output = sum;
            return Output;
        }

        public double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }

        public void SetWeights(params double[] weights)
        {
            //TODO потом убрать
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }

    public enum EnNeuronType
    {
        Input = 0,
        Normal = 1,
        Output = 2
    }
}
