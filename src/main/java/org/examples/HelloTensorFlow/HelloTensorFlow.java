package org.examples.HelloTensorFlow;

import org.tensorflow.ConcreteFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Add;
import org.tensorflow.types.TInt32;

public class HelloTensorFlow {

    public static void main(String[] args) {
        System.out.println("Hello TensorFlow " + TensorFlow.version());

        //tries to create a ConcreteFunction that doubles an integer.
        try (ConcreteFunction doubleInteger = ConcreteFunction.create(HelloTensorFlow::doubleInteger);
             //creates a TensorFlow scalar tensor with the value 10
             TInt32 x = TInt32.scalarOf(10);
             //calls the ConcreteFunction with the input tensor x, which contains the value 10 -> performs the operation
             Tensor dblX = doubleInteger.call(x)) {
            System.out.println(x.getInt() + " doubled is " + ((TInt32) dblX).getInt());
        }
    }

    private static Signature doubleInteger(Ops tf) {
        //creates placeholder for input value
        Placeholder<TInt32> x = tf.placeholder(TInt32.class);
        //uses the TensorFlow add operation to add the placeholder x to itself = doubling it
        Add<TInt32> dblX = tf.math.add(x, x);
        //defines the inputs and outputs which are used in our main method, using a Signature
        return Signature.builder().input("x", x).output("doubleInteger", dblX).build();
    }

}

