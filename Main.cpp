#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <Eigen/Dense>
#include <random>

using namespace Eigen;

constexpr int g_nbTrainingData = 10;
constexpr float g_nbTrainingDataInv = 1.0f / static_cast<float>(g_nbTrainingData);
constexpr int g_nbFeatures = 10;


template <int nbNodes, int nbInputNodes>
class Layer
{
public:
    Layer()
    {
        nodes.setZero();

        std::random_device rd;
        std::mt19937 rng(rd());
        std::normal_distribution<float> dist(0.0, 1.0);

        weights = weights.unaryExpr([&](float x) { return dist(rng) ; });
        biases = biases.unaryExpr([&](float x) { return dist(rng) ; });
    }
    void Print() const {
        std::cout << "Weights:\n" << weights << std::endl;
        std::cout << "Biases:\n" << biases << std::endl;
        std::cout << "Nodes:\n" << nodes << std::endl;
    }
    void FeedForward(const Eigen::Matrix<float, nbInputNodes, g_nbTrainingData>& previousLayer)
    {
        nodes = (weights * previousLayer).colwise() + biases;
        nodes = nodes.unaryExpr([this](float x) { return Sigmoid(x); });
    }

    Eigen::Matrix<float, nbNodes, g_nbTrainingData> nodes;
    Eigen::Matrix<float, nbNodes, nbInputNodes> weights;
    Eigen::Matrix<float, nbNodes, 1> biases;
private:


    // z is a vector of all nodes in a layer
    // g(z) notation for sigmoid
    float Sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

template<int nbNodes, int nbInputNodes>
float Cost(
    const Layer<nbNodes, nbInputNodes>& outputLayer,
    Eigen::Matrix<float, nbNodes, g_nbTrainingData> trainingLabels
)
{
    Eigen::Matrix<float, nbNodes, g_nbTrainingData> y_hat = outputLayer.nodes;

    Eigen::Matrix<float, nbNodes, g_nbTrainingData> losses = -(
        (trainingLabels.array() * y_hat.array().log()) +
        ((1 - trainingLabels.array()) * (1 - y_hat.array()).log())
        );

    Eigen::RowVectorXf summed_losses = losses.colwise().sum();
    summed_losses /= g_nbTrainingData;

    return summed_losses.sum();
}


void backprop_layer_3(const MatrixXf& y_hat, const MatrixXf& Y,
    int m, const MatrixXf& A2, const MatrixXf& W3,
    MatrixXf& dC_dW3, MatrixXf& dC_db3, MatrixXf& dC_dA2) {

    MatrixXf dC_dZ3 = (1.0 / m) * (y_hat - Y);

    dC_dW3 = dC_dZ3 * A2.transpose();

    dC_db3 = dC_dZ3.rowwise().sum();

    dC_dA2 = W3.transpose() * dC_dZ3;
}

void backprop_layer_2(const MatrixXf& propagator_dC_dA2, const MatrixXf& A1,
    const MatrixXf& A2, const MatrixXf& W2,
    MatrixXf& dC_dW2, MatrixXf& dC_db2, MatrixXf& dC_dA1) {

    MatrixXf dA2_dZ2 = A2.array() * (1.0 - A2.array());
    MatrixXf dC_dZ2 = propagator_dC_dA2.array() * dA2_dZ2.array();

    dC_dW2 = dC_dZ2 * A1.transpose();

    dC_db2 = dC_dZ2.rowwise().sum();


    Eigen::MatrixXf transposed = W2.transpose();

    dC_dA1 = transposed * dC_dZ2;
}

void backprop_layer_1(const MatrixXf& propagator_dC_dA1, const MatrixXf& A0,
    const MatrixXf& A1, const MatrixXf& W1,
    MatrixXf& dC_dW1, MatrixXf& dC_db1) {

    MatrixXf dA1_dZ1 = A1.array() * (1.0 - A1.array());
    MatrixXf dC_dZ1 = propagator_dC_dA1.array() * dA1_dZ1.array();

    dC_dW1 = dC_dZ1 * A0.transpose();

    dC_db1 = dC_dZ1.rowwise().sum();
}

int main()
{

    const int nbLayers = 3;
    constexpr std::array<int, 1 + nbLayers> n[] = { g_nbFeatures, 4, 4, g_nbFeatures };


    Eigen::Matrix<float, g_nbTrainingData, g_nbFeatures> trainingData;
    trainingData.setZero();


    for (int i = 0; i < 10; i++)
    {
        trainingData(i, i) = 1.0f;
    }

    Eigen::Matrix<float, g_nbFeatures, g_nbTrainingData> A0 = trainingData.transpose();


    Eigen::Matrix<float, n->at(nbLayers), g_nbTrainingData> trainingLables;
    trainingLables.setZero();
    for (int i = 0; i < 10; i++)
    {
        trainingLables(static_cast<unsigned int>(i + 1) % 10, i) = 1.0f;
    }
    float learningRate = 0.1f;

    // In general, we will represent the vectorized layer of activations across all training samples for a layer l as A^ [l].
    // These vectorized activation layers will always have m(number of training samples) columns, 
    // and their number of rows is equal to the number of nodes in that layer, n^ [l].

    // m (number of training samples)
    // n (nodes in layer)

    // The final layer output A^[L] has dimensions n^[L] x m,
    Layer<n->at(1), g_nbFeatures> A1; // A^ [l]
    Layer<n->at(2), n->at(1)> A2; // A^ [2]
    Layer<n->at(nbLayers), n->at(2)> A3; // A^ [3]
    for (size_t i = 0; i < 1000000; i++)
    {
        A1.FeedForward(A0);
        A2.FeedForward(A1.nodes);
        A3.FeedForward(A2.nodes);
        
        if (i % 10000 == 0)
        {
            std::cout << "Cost: " << Cost(A3, trainingLables) << std::endl;
        }
        
        MatrixXf dC_dW3, dC_db3, dC_dA2;
        backprop_layer_3(A3.nodes, trainingLables, g_nbTrainingData, A2.nodes, A3.weights, dC_dW3, dC_db3, dC_dA2);
        
        MatrixXf dC_dW2, dC_db2, dC_dA1;
        backprop_layer_2(dC_dA2, A1.nodes, A2.nodes, A2.weights, dC_dW2, dC_db2, dC_dA1);
        
        MatrixXf dC_dW1, dC_db1;
        backprop_layer_1(dC_dA1, A0, A1.nodes, A1.weights, dC_dW1, dC_db1);
        
        A3.weights = A3.weights - (learningRate * dC_dW3);
        A2.weights = A2.weights - (learningRate * dC_dW2);
        A1.weights = A1.weights - (learningRate * dC_dW1);
        A3.biases = A3.biases - (learningRate * dC_db3);
        A2.biases = A2.biases - (learningRate * dC_db2);
        A1.biases = A1.biases - (learningRate * dC_db1);
    }


    //for (int i = 0; i < g_nbTrainingData; i++)
    //{
    //    MatrixXf inputData(1, 2);
    //    inputData << trainingData(i, 0), trainingData(i, 1);
    //
    //    A1.FeedForward(inputData);
    //    A2.FeedForward(A1.nodes);
    //    A3.FeedForward(A2.nodes);
    //
    //    std::cout << "Prediction: " << A3.nodes(0, 0) << " should be: " << trainingLables(0, i) << std::endl;
    //}
    //


    std::cout << "Enter a number between 0 and 10 and ai will predict the next one." << std::endl;

    while (true)
    {
        std::string input;
        std::cin >> input;

        int inputNumber = std::stoi(input);

        MatrixXf inputData(1, g_nbFeatures);
        inputData.setZero();
        inputData(0, inputNumber) = 1.0f;


        A1.FeedForward(inputData);
        A2.FeedForward(A1.nodes);
        A3.FeedForward(A2.nodes);

        float highestValue = -1.0f;
        int indexOfHighest = -1;

        for (int i = 0; i < g_nbFeatures; i++) 
        {
            if (A3.nodes(i, 0) > highestValue) 
            {
                highestValue = A3.nodes(i, 0); 
                indexOfHighest = i;
            }
        }
        std::cout << "Next number is: " << indexOfHighest << " with " << std::to_string(highestValue * 100.0f) << "% certanty" << std::endl;
    }
    
   

    return 0;
}
