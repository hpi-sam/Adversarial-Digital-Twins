package de.mdelab.morisia.comparch.simulator.impl;

import java.util.List;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ComparchSimulatorException;
import de.mdelab.morisia.comparch.simulator.Injection;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;

/**
 * Controls the injection of issues into the model using the
 * {@link InjectionStrategy}.
 * 
 * @author thomas vogel
 * @version 0.30
 *
 */
public class InjectionController {

	private final static Logger LOGGER = Logger
			.getLogger(InjectionController.class.getName());

	// params
	private List<Injector<? extends ArchitecturalElement>> injectors;
	private InjectionStrategy strategy;

	/**
	 * Creates the controller.
	 * 
	 * @param injectors
	 *            the list of injectors used by the simulator.
	 */
	public InjectionController(
			List<Injector<? extends ArchitecturalElement>> injectors) {
		this.injectors = injectors;
	}

	/**
	 * Sets the injection strategy to this controller.
	 * 
	 * @param strategy
	 *            the strategy to be used by this controller.
	 */
	public void setInjectionStrategy(InjectionStrategy strategy) {
		this.strategy = strategy;
	}

	/**
	 * Retrieves the injector for the given issue type and target model element.
	 * 
	 * @param issueType
	 *            the type of issue
	 * @param target
	 *            the target model element
	 * @return the corresponding injector.
	 */
	@SuppressWarnings("unchecked")
	private <T extends ArchitecturalElement> Injector<T> getInjector(
			IssueType issueType, T target) {
		Injector<T> injector = null;
		for (Injector<? extends ArchitecturalElement> i : this.injectors) {
			if (i.getIssueType() == issueType) {
				injector = (Injector<T>) i;
			}
		}
		if (injector == null) {
			throw new ComparchSimulatorException(
					"There is no injector available for the issue type "
							+ issueType);
		} else {
			return injector;
		}
	}

	/**
	 * Injects the issues into the model for one simulation run.
	 * 
	 * @param currentRun
	 *            the number of the current run of the simulator.
	 */
	public void injectIssues(int currentRun) {

		List<Injection<? extends ArchitecturalElement>> injections = this.strategy
				.getInjections(currentRun);

		if (injections == null) {
			throw new IllegalArgumentException(
					"The injection strategy must not return null for the getInjections method.");
		}

		if (injections.size() > 0) {
			LOGGER.info("Trying to inject " + injections.size()
					+ " issue(s) in this run.");
		} else {
			LOGGER.info("Injecting no issue(s) in this run.");
		}

		for (int i = 0; i < injections.size(); i++) {
			Injection<? extends ArchitecturalElement> injection = injections
					.get(i);
			IssueType issueType = injection.getIssueType();
			ArchitecturalElement target = injection.getTarget();
			boolean success = false;
			Tenant tenant = null;

			// CF4 need not to be handled here since it is a result of the
			// consecutive occurrence of other CFs.
			if (target instanceof Component) {
				// CF1, CF3
				Component component = (Component) target;
				Injector<Component> injector = this.getInjector(issueType,
						component);
				tenant = component.getTenant();
				success = injector.inject(component);
			} else if (target instanceof ProvidedInterface) {
				// CF2, NO_CF2
				ProvidedInterface providedInterface = (ProvidedInterface) target;
				Injector<ProvidedInterface> injector = this.getInjector(
						issueType, providedInterface);
				tenant = providedInterface.getComponent().getTenant();
				success = injector.inject(providedInterface);
			} else if (target instanceof Connector) {
				// CF5
				Connector connector = (Connector) target;
				Injector<Connector> injector = this.getInjector(issueType,
						connector);
				tenant = connector.getSource().getComponent().getTenant();
				success = injector.inject(connector);
			} else if (target instanceof Tenant) {
				// PI1, PI2, PI3
				tenant = (Tenant) target;
				Injector<Tenant> injector = this.getInjector(issueType, tenant);
				success = injector.inject(tenant);
			}
			injection.setSuccess(success);
			if (success) {
				LOGGER.info("The simulator successfully injected a "
						+ issueType.toString() + " in tenant \""
						+ tenant.getName() + "\".");
			} else {
				LOGGER.info("The simulator cannot inject a "
						+ issueType.toString() + " in tenant \""
						+ tenant.getName() + "\".");
			}
		}
		// done with the run. Notify the strategy about the success.
		this.strategy.notifyAboutSuccess(injections);
	}
}
