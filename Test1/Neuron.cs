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
        public List<double> Inputs { get; }
        public EnNeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }

        public Neuron(int inputCount, EnNeuronType neuronType = EnNeuronType.Normal)
        {
            NeuronType = neuronType;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitWeightsRandomValues(inputCount);
        }

        private void InitWeightsRandomValues(int inputCount)
        {
            Random rnd = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(rnd.NextDouble());
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            double sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
                sum += inputs[i] * Weights[i];
            }
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

        public double SigmoidDx(double x)
        {
            double sigmoid = Sigmoid(x);
            double result = sigmoid / (1 - sigmoid);
            return result;
        }

        public void SetWeights(params double[] weights)
        {
            //TODO потом убрать
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        public void Learn(double error, double learningRaid)
        {
            if (NeuronType == EnNeuronType.Input)
                return;

            Delta = error * SigmoidDx(Output);
            for (int x = 0; x < Weights.Count; x++)
            {
                double weight = Weights[x];
                double input = Inputs[x];

                double newWeight = weight - input * Delta * learningRaid;
                Weights[x] = newWeight;
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
