package de.mdelab.morisia.comparch.simulator;

import java.util.List;

import de.mdelab.morisia.comparch.ArchitecturalElement;

/**
 * A strategy defining the injection of issues. A strategy specifies a scenario
 * that the simulator will follow.
 * 
 * @author thomas vogel
 * @version 0.30
 */
public interface InjectionStrategy {

	/**
	 * For each simulation run, the simulator consults the strategy to obtain
	 * the injections to be performed for this run.
	 * 
	 * @param runCount
	 *            the number of the current run.
	 * @return the list of injections to be performed.
	 */
	public List<Injection<? extends ArchitecturalElement>> getInjections(
			int runCount );

	/**
	 * The simulator notifies the strategy immediately after one simulation run
	 * about the success of the injections.
	 * 
	 * @param injections
	 *            the list of injections defined by the strategy and whose
	 *            individual success attributes are set (see
	 *            {@link Injection#isSuccess()}).
	 */
	public void notifyAboutSuccess(
			List<Injection<? extends ArchitecturalElement>> injections);

}
