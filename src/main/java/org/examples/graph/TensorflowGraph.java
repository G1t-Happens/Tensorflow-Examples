package org.examples.graph;

import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.TFloat64;


public class TensorflowGraph {


    public static void main(String[] args) {
        // Create a TensorFlow graph
        Graph graph = TensorflowGraph.createGraph();

        // Run the graph with input values 3.0 and 6.0
        Double result = TensorflowGraph.runGraph(graph, 3.0, 6.0);

        // Print the result to the console
        System.out.println(result);

        // Close the TensorFlow graph
        graph.close();
    }

    // Creates a TensorFlow graph with constants, placeholders, and operations
    public static Graph createGraph() {
        // Create a new TensorFlow graph
        Graph graph = new Graph();

        // Create an Ops object to build operations in the graph
        Ops ops = Ops.create(graph);

        // Define two constant values, a and b
        Constant<TFloat64> a = ops.constant(3.0);
        Constant<TFloat64> b = ops.constant(2.0);
        Constant<TFloat64> c = ops.constant(4.0);

        // Define two placeholders, x and y, for input values
        Placeholder<TFloat64> x = ops.placeholder(TFloat64.class);
        Placeholder<TFloat64> y = ops.placeholder(TFloat64.class);

        // Define operations to calculate ax and by
        Operand<TFloat64> ax = ops.math.mul(a, x); //3.0 * 3.0 = 9.0
        Operand<TFloat64> by = ops.math.mul(b, y); //2.0 * 6.0 = 12.0

        // Define an operation to add ax and by to get the final result z
        Operand<TFloat64> z = ops.math.add(ax, by); // 9.0 + 12.0 = 21

        // Define an operation to divide z with  c to get the final result q
        Operand<TFloat64> q = ops.math.div(z, c); // 21 / 4 = 5.25


        // Return the created graph
        return graph;
    }

    // Runs the TensorFlow graph with input values x and y and returns the result
    public static Double runGraph(Graph graph, Double x, Double y) {
        try (Session session = new Session(graph)) {
            // Create TensorFlow tensors for input values x and y
            TFloat64 xTensor = TFloat64.scalarOf(x);
            TFloat64 yTensor = TFloat64.scalarOf(y);

            // Run the graph, fetch the "Add" operation, and feed the input values
            Tensor z = session.runner()
                    .fetch("Div") // Name of the addition operation in the graph
                    .feed("Placeholder:0", xTensor) // Feed input value x to the Placeholder
                    .feed("Placeholder_1:0", yTensor) // Feed input value y to the Placeholder
                    .run()
                    .get(0);

            // Get the result as a Double and return it
            return ((TFloat64) z).getDouble();
        }
    }

}

