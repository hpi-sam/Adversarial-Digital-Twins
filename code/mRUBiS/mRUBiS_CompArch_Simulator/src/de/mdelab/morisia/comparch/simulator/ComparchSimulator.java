package de.mdelab.morisia.comparch.simulator;

import java.util.List;

import de.mdelab.morisia.comparch.ArchitecturalElement;

/**
 * Interface to use the simulator.
 * 
 * If the simulator has been obtained from the factory with the
 * {@link Capability#BLANK}, then it must be configured by adding
 * {@link Injector}s (see {@link #addInjector(Injector)}) and {@link Analyzer}s
 * (see {@link #addAnalyzer(Analyzer)}).
 * 
 * If the simulator has the capabilities {@link Capability#SELF_REPAIR},
 * {@link Capability#SELF_OPTIMIZATION}, or {@link Capability#SELF_MANAGEMENT},
 * then the corresponding injectors and analyzers have been automatically added.
 * 
 * Finally, an {@link InjectionStrategy} can be set to the simulator to specify
 * a scenario for the simulation. See method
 * {@link #setInjectionStrategy(InjectionStrategy)}. If no strategy is set, then
 * the simulator uses its default strategy (random injection of issues).
 * 
 * Then, the simulator can be used by invoking {@link #validate()} to analyze
 * the model and {@link #injectIssues()} to inject issues into the model. These
 * methods respectively invoke {@code Injector}s according to the
 * {@link InjectionStrategy} as well as the {@code Analyzer}s.
 * 
 * Finally, {@link #isSimulationCompleted()} is used to check whether the
 * simulation has been completed or not.
 * 
 * @author thomas vogel
 * @version 0.30
 *
 */
public interface ComparchSimulator {

	/**
	 * The singleton factory instance to obtain an instance of the
	 * {@code ComparchSimulator}.
	 */
	public final static ComparchSimulatorFactory FACTORY = new ComparchSimulatorFactory();

	/**
	 * Sets the injection strategy that replaces the default strategy of the
	 * simulator.
	 * 
	 * @param strategy
	 *            the injection strategy
	 */
	public void setInjectionStrategy(InjectionStrategy strategy);

	/**
	 * @return a unmodifiable list of the injectors used by the simulator.
	 */
	public List<Injector<? extends ArchitecturalElement>> getInjectors();

	/**
	 * @return a list of {@link IssueType}s that are supported by the simulator
	 *         according to its configuration and capability.
	 */
	public IssueType[] getSupportedIssueTypes();

	/**
	 * Returns <code>true</code> if the simulation has been completed, else
	 * <code>false</code>.
	 * 
	 * @return <code>true</code> if the simulation has been completed, else
	 *         <code>false</code>.
	 */
	public boolean isSimulationCompleted();

	/**
	 * Adds the given injector to the simulator.
	 * 
	 * @param injector
	 *            the injector to be added.
	 */
	public void addInjector(Injector<? extends ArchitecturalElement> injector);

	/**
	 * Adds the given analyzer to the simulator.
	 * 
	 * @param analyzer
	 *            the analyzer to be added.
	 */
	public void addAnalyzer(Analyzer analyzer);

	/**
	 * Validates the model using the analyzers.
	 * 
	 * @return the number of failures identified by the validation.
	 */
	public int validate();

	/**
	 * Injects issues into the model using the injectors according to the
	 * {@link InjectionStrategy}.
	 * 
	 * The issues injected by one invocation of this method should be handled by
	 * one run of the feedback loop. Hence, one invocation of this method
	 * corresponds to one run of the simulator.
	 */
	public void injectIssues();

}
