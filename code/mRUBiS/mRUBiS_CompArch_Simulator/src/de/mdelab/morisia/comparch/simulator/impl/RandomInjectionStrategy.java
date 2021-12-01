package de.mdelab.morisia.comparch.simulator.impl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RequiredInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injection;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.IssueType;

/**
 * The default strategy defining the injection of issues.
 * 
 * This strategy performs random injections while considering that in one run of
 * the simulator, a tenant in the model can only be affected at most once by an
 * issue. Hence, there are not more than one issue injected to the same tenant
 * within one run of the tenant.
 * 
 * @author thomas vogel
 * @version 0.30
 *
 */
public class RandomInjectionStrategy implements InjectionStrategy {

	private final static Logger LOGGER = Logger
			.getLogger(RandomInjectionStrategy.class.getName());

	private IssueType[] issueTypes;
	private Architecture eArchitecture;
	private int lowerBoundOfConcurrentInjections = 0;
	private int upperBoundOfConcurrentInjections = 0;
	private Map<IssueType, Integer> numberOfExecutions;
	private List<Tenant> blacklist;

	/**
	 * Creates the strategy.
	 * 
	 * @param issueTypes
	 *            the issue types supported by the simulator.
	 * @param eArchitecture
	 *            the root element of the model.
	 */
	public RandomInjectionStrategy(IssueType[] issueTypes,
			Architecture eArchitecture) {
		this.issueTypes = issueTypes;
		LOGGER.info("Default injection strategy takes the following issues into account: "
				+ Arrays.toString(this.issueTypes));

		this.eArchitecture = eArchitecture;
		this.numberOfExecutions = new HashMap<IssueType, Integer>();
		this.blacklist = new LinkedList<Tenant>();

		int numberOftenants = this.eArchitecture.getTenants().size();
		// 25% of the tenants can be affected in one run
		this.upperBoundOfConcurrentInjections = (int) Math
				.round(numberOftenants * 0.25);
		if (this.upperBoundOfConcurrentInjections <= 1) {
			this.upperBoundOfConcurrentInjections = this.eArchitecture
					.getTenants().size();
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<Injection<? extends ArchitecturalElement>> getInjections(
			int runCount) {
		List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();
		int concurrentInjections = SimulatorUtil.randInt(
				this.lowerBoundOfConcurrentInjections,
				this.upperBoundOfConcurrentInjections);
		for (int i = 0; i < concurrentInjections; i++) {
			// select random issue
			IssueType issueType = this.selectIssueType();
			if (issueType == IssueType.CF1 || issueType == IssueType.CF3) {
				// select random component
				Component component = this.selectRandomComponent();
				if (component != null) {
					injections.add(new Injection<Component>(issueType,
							component));
				}
			} else if (issueType == IssueType.CF2
					|| issueType == IssueType.NO_CF2) {
				// select random ProvidedInterface
				ProvidedInterface providedInterface = this
						.selectRandomUsedProvidedInterface();
				if (providedInterface != null) {
					injections.add(new Injection<ProvidedInterface>(issueType,
							providedInterface));
				}
			} else if (issueType == IssueType.CF4) {
				// select random Connector
				Connector connector = this.selectRandomConnector();
				if (connector != null) {
					injections.add(new Injection<Connector>(issueType,
							connector));
				}
			} else if (issueType == IssueType.PI1 || issueType == IssueType.PI2
					|| issueType == IssueType.PI3) {
				// select random Tenant
				Tenant tenant = this.selectRandomTenant();
				if (tenant != null) {
					injections.add(new Injection<Tenant>(issueType, tenant));
				}
			}

		}
		this.blacklist.clear();
		return injections;
	}

	/**
	 * @return a random tenant of the model.
	 */
	private Tenant selectRandomTenant() {
		List<Tenant> allTenants = new LinkedList<Tenant>();
		allTenants.addAll(this.eArchitecture.getTenants());
		allTenants.removeAll(this.blacklist);

		if (allTenants.size() > 0) {
			int tenantNumber = SimulatorUtil.randInt(0, allTenants.size() - 1);
			Tenant tenant = allTenants.get(tenantNumber);
			this.blacklist.add(tenant);
			return tenant;
		} else {
			return null;
		}
	}

	/**
	 * @return a random component of the model.
	 */
	private Component selectRandomComponent() {
		Tenant tenant = this.selectRandomTenant();
		if (tenant != null) {
			Component component = null;
			while (component == null) {
				int componentNumber = SimulatorUtil.randInt(0, tenant
						.getComponents().size() - 1);
				component = tenant.getComponents().get(componentNumber);
				if (component.getType().getName()
						.equals("Future Sales Item Filter")) {
					// last filter of the pipe should bot be affected by an
					// issue.
					component = null;
				}
			}
			return component;
		} else {
			return null;
		}
	}

	/**
	 * @return a random provided interface of the model, which is used by a
	 *         required interface.
	 */
	private ProvidedInterface selectRandomUsedProvidedInterface() {
		List<ProvidedInterface> usedInterfaces = new LinkedList<ProvidedInterface>();
		int tries = 0;

		while (usedInterfaces.size() == 0 || tries < 50) {
			tries++;
			Component component = this.selectRandomComponent();
			if (component != null) {
				for (ProvidedInterface pi : component.getProvidedInterfaces()) {
					if (pi.getConnectors().size() > 0) {
						usedInterfaces.add(pi);
					}
				}
			} else {
				break;
			}
		}
		if (usedInterfaces.size() > 0) {
			return usedInterfaces.get(SimulatorUtil.randInt(0,
					usedInterfaces.size() - 1));
		} else {
			return null;
		}
	}

	/**
	 * @return a random connector of the model.
	 */
	private Connector selectRandomConnector() {
		Connector connector = null;
		int tries = 0;
		while (connector == null || tries > 25) {
			tries++;
			Component component = this.selectRandomComponent();
			if (component != null) {
				for (RequiredInterface ri : component.getRequiredInterfaces()) {
					Connector c = ri.getConnector();
					if (c != null && c.getTarget() != null) {
						connector = c;
						break;
					}
				}
			} else {
				break;
			}
		}
		return connector;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void notifyAboutSuccess(
			List<Injection<? extends ArchitecturalElement>> injections) {
		for (Injection<? extends ArchitecturalElement> i : injections) {
			if (!i.isSuccess()) {
				LOGGER.info("Simulator could not successfully inject a "
						+ i.getIssueType() + " issue to element "
						+ i.getTarget());
			} else {
				IssueType issueType = i.getIssueType();
				Integer execution = this.numberOfExecutions.get(issueType);
				if (execution == null) {
					this.numberOfExecutions.put(issueType, new Integer(1));
				} else {
					this.numberOfExecutions.put(issueType, new Integer(
							execution.intValue() + 1));
				}
			}
		}

	}

	/**
	 * Selects an issue type randomly and considers balancing the number of
	 * using the individual issue types.
	 * 
	 * @return the selected issue type
	 */
	private IssueType selectIssueType() {
		IssueType issueType = null;

		// balance use of the different injectors
		int min = Integer.MAX_VALUE;
		for (IssueType i : this.issueTypes) {
			Integer useOfI = this.numberOfExecutions.get(i);
			if (useOfI == null) {
				// at least one injector has not been uised at all
				min = 0;
				break;
			} else {
				if (useOfI < min) {
					min = useOfI;
				}
			}
		}

		List<IssueType> minInjectedIssue = new LinkedList<IssueType>();
		for (IssueType i : this.issueTypes) {
			Integer useOfI = this.numberOfExecutions.get(i);
			if (useOfI == null) {
				useOfI = 0;
			}
			if (min == useOfI) {
				minInjectedIssue.add(i);
			}
		}

		int selectedIssue = SimulatorUtil.randInt(0,
				minInjectedIssue.size() - 1);
		issueType = minInjectedIssue.get(selectedIssue);

		return issueType;
	}

}
