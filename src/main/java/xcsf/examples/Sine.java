package xcsf.examples;

import xcsf.Function.SimpleFunction;

/**
 * Implements a sine function.
 * 
 * @author Patrick O. Stalph, Martin V. Butz
 */
public class Sine extends SimpleFunction {

    /**
     * Default constructor.
     * 
     * @param scale
     *            the scaling factor
     * @param modifier
     *            the function modifier
     * @param noiseDeviation
     *            the (gaussian) noise deviation
     * @param dim
     *            the input dimension of the function
     */
    public Sine(double scale, double modifier, double noiseDeviation, int dim) {
        super(scale, modifier, noiseDeviation, dim);
    }

    /*
     * (non-Javadoc)
     * 
     * @see xcsf.Function.SimpleFunction#evaluate()
     */
    protected double evaluate() {
        double sum = 0.0;
        for (int i = 0; i < super.dim; i++) {
            sum += input[i];
        }
        return Math.sin(super.modifier * Math.PI * sum);
    }
}
