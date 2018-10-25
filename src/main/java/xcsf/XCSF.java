package xcsf;

import java.util.Collection;
import java.util.Vector;

import xcsf.classifier.PredictionLinearRLS;
import xcsf.classifier.PredictionQuadraticRLS;

/**
 * The main XCSF class coordinates the learning process and listeners. A typical
 * use case looks as simple as this:
 * 
 * <pre>
 * // load your function, implementing the xcsf.functions.Function interface
 * Function f = new Sine(1, 4, 0, 2);
 * // load XCSF settings from a file
 * XCSFConstants.load(&quot;xcsf.ini&quot;);
 * // create the XCSF instance
 * XCSF xcsf = new XCSF(f);
 * // add some listeners for performance monitoring or visualization
 * xcsf.addListener(new PerformanceGUI(true));
 * // start the experiments
 * xcsf.runExperiments();
 * </pre>
 * 
 * @author Patrick O. Stalph, Martin V. Butz
 * @see xcsf.Function
 * @see xcsf.XCSFListener
 */
public class XCSF {

    // the function to approximate
    private Function function;
    // list of registered listeners
    private Vector<XCSFListener> listeners = new Vector<XCSFListener>();
    // keeps track of learning performance, including avg. error and pop. size
    private PerformanceEvaluator performanceEvaluator;

    /**
     * XCSF constructor creates the xcsf instance using the given
     * <code>function</code>.
     * 
     * @param function
     *            the function to be learned
     */
    public XCSF(Function function) {
        this.function = function;
    }

    /**
     * Convenience constructor that also loads XCSF's settings from the given
     * <code>settingsFilename</code>.
     * 
     * @param function
     *            the function to be learned
     * @param settingsFilename
     *            the filename of XCSF's settings file to be loaded by
     *            {@link XCSFConstants#load(String)}.
     */
    public XCSF(Function function, String settingsFilename) {
        this(function);
        XCSFConstants.load(settingsFilename);
    }

    /**
     * This method starts all experiments by calling
     * {@link #runSingleExperiment()} for
     * {@link XCSFConstants#numberOfExperiments} times. Furthermore, in verbose
     * mode some informations about the progress are printed to
     * <code>System.out</code>.
     */
    public void runExperiments() {
        XCSFUtils.println("");
        this.performanceEvaluator = new PerformanceEvaluator();
        // run several single experiments
        for (int exp = 0; exp < XCSFConstants.numberOfExperiments; exp++) {
            // listeners: indicate next experiment
            for (XCSFListener l : listeners) {
                l.nextExperiment(exp, this.function.toString());
            }

            if (XCSFConstants.numberOfExperiments > 1) {
                XCSFUtils.print(" > " + this.function.toString()
                        + ", Experiment " + (exp + 1) + "/"
                        + XCSFConstants.numberOfExperiments, 60);
            } else {
                XCSFUtils.println("Running on Function '"
                        + this.function.toString() + "'...");
            }
            long time = System.currentTimeMillis();

            // start xcsf single run
            this.runSingleExperiment();

            time = (System.currentTimeMillis() - time) / 1000;
            XCSFUtils.println("done in " + (int) (time / 60) + "m "
                    + (time % 60) + "s");
        }
    }

    /**
     * This method starts one experiment for
     * {@link XCSFConstants#maxLearningIterations} and returns the final
     * population.
     * 
     * @return the final population
     */
    public Population runSingleExperiment() {
        return runSingleExperiment(new Population());
    }

    /**
     * This method starts one experiment for
     * {@link XCSFConstants#maxLearningIterations} using the given inital
     * <code>population</code>.
     * 
     * @param population
     *            the initial population
     * @return the final population, which is a reference to the given
     *         <tt>population</tt>
     */
    public Population runSingleExperiment(Population population) {
        if (this.performanceEvaluator == null) {
            this.performanceEvaluator = new PerformanceEvaluator();
        }
        MatchSet matchSet = new MatchSet(XCSFConstants.doNumClosestMatch,
                XCSFConstants.multiThreading);
        EvolutionaryComp evolutionaryComponent = new EvolutionaryComp();
        this.performanceEvaluator.nextExperiment();

        // -----[ main loop ]-----
        for (int iteration = 1; iteration <= XCSFConstants.maxLearningIterations; iteration++) {
            // 1) get next problem instance
            StateDescriptor state = this.function.nextProblemInstance();
            // 2) match & cover if necessary
            matchSet.match(state, population);// most computational time here
            matchSet.ensureStateCoverage(population, iteration);
            // 3) evaluate performance
            double[] functionPrediction = matchSet.getWeightedPrediction();
            double[] functionValue = this.function.getNoiselessFunctionValue();
            if (functionValue == null) {
                // no noiseless value for performance tracking available
                functionValue = state.getOutput();
            }
            this.performanceEvaluator.evaluate(population, matchSet, iteration,
                    functionValue, functionPrediction);
            // 4) update matching classifiers
            matchSet.updateClassifiers();
            // 5) evolution
            evolutionaryComponent
                    .evolve(population, matchSet, state, iteration);

            // inform listeners
            if (!listeners.isEmpty()) {
                double[][] performance = this.performanceEvaluator
                        .getCurrentExperimentPerformance();
                for (XCSFListener l : listeners) {
                    l.stateChanged(iteration, population, matchSet, state,
                            performance);
                }
            }

            // reset rls prediction at next iteration?
            if ((XCSFConstants.predictionType
                    .equalsIgnoreCase(PredictionLinearRLS.class.getName()) || XCSFConstants.predictionType
                    .equalsIgnoreCase(PredictionQuadraticRLS.class.getName()))
                    && iteration + 1 == (int) (XCSFConstants.resetRLSPredictionsAfterSteps * XCSFConstants.maxLearningIterations)) {
                for (int i = 0; i < population.size; i++) {
                    ((PredictionLinearRLS) population.elements[i]
                            .getPrediction()).resetGainMatrix();
                }
            }

            // start compaction at next iteration?
            if (iteration + 1 == (int) (XCSFConstants.startCompaction * XCSFConstants.maxLearningIterations)) {
                evolutionaryComponent.setCondensation(true);
                if (XCSFConstants.compactionType % 2 == 1) {
                    // type 1 & 3
                    matchSet.setNumClosestMatching(true);
                }
                if (XCSFConstants.compactionType >= 2) {
                    // type 2 & 3
                    population.applyGreedyCompaction();
                }
            }
        } // ---[ end loop ]------

        // make sure that child threads are closed.
        try {
            matchSet.shutDownThreads();
            matchSet = null;
        } catch (Throwable e) {
            // ignore
        }
        return population;
    }

    /**
     * Returns mean and variance values of the final iteration represented in a
     * tab-separated String.
     * 
     * @return the performance of the final iteration
     */
    public String getFinalPerformance() {
        return this.performanceEvaluator.getAvgFinalPerformance();
    }

    /**
     * Writes the performance to the given <code>filename</code>.
     * 
     * @param filename
     *            the file to write to
     */
    public void writePerformance(String filename) {
        this.performanceEvaluator.writeAvgPerformance(filename);
    }

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    public String toString() {
        StringBuffer sb = new StringBuffer(100);
        sb.append("XCSF on ");
        sb.append(this.function.toString());
        sb.append(" in ");
        sb.append(this.function.getConditionInputDimension());
        sb.append("D");
        sb.append(", settings:");
        sb.append(System.getProperty("line.separator"));
        sb.append(" * max ");
        sb.append(XCSFConstants.maxPopSize);
        sb.append(" CLs for ");
        sb.append(XCSFConstants.maxLearningIterations);
        sb.append(" iterations, target error=");
        sb.append(XCSFConstants.epsilon_0);
        sb.append(System.getProperty("line.separator"));
        sb.append(" * ");
        sb.append(XCSFConstants.conditionType
                .substring(1 + XCSFConstants.conditionType.lastIndexOf('.')));
        sb.append("/");
        sb.append(XCSFConstants.predictionType
                .substring(1 + XCSFConstants.predictionType.lastIndexOf('.')));
        if (XCSFConstants.multiThreading
                && Runtime.getRuntime().availableProcessors() > 1) {
            sb.append(", up to ");
            sb.append(Runtime.getRuntime().availableProcessors());
            sb.append(" threads");
        }
        return sb.toString();
    }

    /**
     * Registers the given listener. Listeners are informed about changes at the
     * end of every iteration in XCSF.
     * 
     * @param listener
     *            the listener to register
     * @see XCSFListener#stateChanged(int, Population, MatchSet,
     *      StateDescriptor, double[][])
     */
    public void addListener(XCSFListener listener) {
        for (XCSFListener el : listeners) {
            if (el.getClass().equals(listener.getClass())) {
                System.err.println("Listener of Class '" + el.getClass()
                        + "' already registered - ignoring " + listener);
                return;
            }
        }
        if (!listeners.contains(listener))
            listeners.add(listener);
    }

    /**
     * Convenience method to add several listeners at once. This can be useful,
     * when various experiments are run and the XCSF instance has to be
     * re-created, but the listeners (e.g. visualizations) remain the same.
     * 
     * @param listener
     *            a collection (list or set) of listeners
     */
    public void addListeners(Collection<XCSFListener> listener) {
        for (XCSFListener l : listener) {
            addListener(l);
        }
    }

    /**
     * Removes the given listener (object reference).
     * 
     * @param l
     *            The listener to remove.
     */
    public void removeListener(XCSFListener l) {
        listeners.remove(l);
    }

    /**
     * Remove all listeners.
     */
    public void clearListeners() {
        listeners.clear();
    }
}
