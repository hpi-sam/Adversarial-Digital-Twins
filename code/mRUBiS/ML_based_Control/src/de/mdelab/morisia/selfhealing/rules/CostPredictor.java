package de.mdelab.morisia.selfhealing.rules;

import java.util.logging.Logger;

import de.mdelab.morisia.comparch.AddReplica;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.HwRedeployComponent;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.LwRedeployComponent;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RecreateConnector;
import de.mdelab.morisia.comparch.RemoveReplica;
import de.mdelab.morisia.comparch.ReplaceComponent;
import de.mdelab.morisia.comparch.RestartComponent;
import de.mdelab.morisia.comparch.Rule;

public class CostPredictor {

	private final static Logger LOGGER = Logger.getLogger(CostPredictor.class
			.getName());

	// Based on measurements/experiments!
	private static double deployComponent_seconds = 1.17;
	private static double configComponent_seconds = 0.16;
	private static double startComponent_seconds = 0.24;
	private static double recreateConnector_seconds = 0.03;
	private static double stopComponent_seconds = 0.11;
	private static double undeployComponent_seconds = 0.12;
	private static double  AddReplica_seconds=0.5;
	private static double  RemoveReplica_seconds=0.6;

	/**
	 * Predicts the costs of each rule handling the given issue.
	 * 
	 * @param issue
	 *            the given issue..
	 */
	public static void predictCosts(Issue issue) {
		Component affectedComponent = issue.getAffectedComponent();
		// ComponentType affectedComponentType = affectedComponent.getType();
		for (Rule rule : issue.getHandledBy()) {
			double exec = 0;
			if (rule instanceof RestartComponent) {
				exec = stopComponent_seconds + startComponent_seconds;
			} else if (rule instanceof LwRedeployComponent) {
				if (affectedComponent.getState() == ComponentLifeCycle.NOT_SUPPORTED) {
					// CF1
					exec = undeployComponent_seconds + deployComponent_seconds
							+ startComponent_seconds;
				} else if (affectedComponent.getState() == ComponentLifeCycle.STARTED) {
					// CF2
					exec = stopComponent_seconds + undeployComponent_seconds
							+ deployComponent_seconds + startComponent_seconds;
				}
			} else if (rule instanceof HwRedeployComponent) {
				exec = deployComponent_seconds
						+ configComponent_seconds
						+ (affectedComponent.getRequiredInterfaces().size() * recreateConnector_seconds)
						+ startComponent_seconds
						+ (getNumberOfUsingComponent(affectedComponent) * recreateConnector_seconds);
				// skip costs of removing the old component since the utility
				// increase by the new component is enabled already before
				// removing the old component. Before the removal, the old
				// component is still in the system but not used anymore since
				// the new component is used instead.
			} else if (rule instanceof ReplaceComponent) {
				ReplaceComponent replaceComponentRule = (ReplaceComponent) rule;
				ComponentType alternative = replaceComponentRule
						.getAlternativeComponentType();
				double reliability =1;
				if (alternative != null) {
				//double	 
					reliability = alternative.getReliability();}
				double inverseReliability = 1 / reliability;
				exec = inverseReliability
						* (deployComponent_seconds
								+ configComponent_seconds
								+ (affectedComponent.getRequiredInterfaces()
										.size() * recreateConnector_seconds)
								+ startComponent_seconds + (getNumberOfUsingComponent(affectedComponent) * recreateConnector_seconds));
				// skip costs of removing the old component since the utility
				// increase by the new component is enabled already before
				// removing the old component. Before the removal, the old
				// component is still in the system but not used anymore since
				// the new component is used instead.
			} else if (rule instanceof RecreateConnector) {
				exec = recreateConnector_seconds;
			}
			else if (rule instanceof AddReplica) {
				exec = AddReplica_seconds;
			}else if (rule instanceof RemoveReplica) {
				exec = RemoveReplica_seconds;
			}

			// TODO: how to predict the costs? Does this makes sense?
			rule.setCosts(exec);
			LOGGER.info("Predicted costs of " + exec + " for the rule " + rule
					+ " handling the issue " + issue);

		}

	}

	private static int getNumberOfUsingComponent(Component component) {
		int users = 0;
		for (ProvidedInterface pi : component.getProvidedInterfaces()) {
			users = users + pi.getConnectors().size();
		}
		return users;
	}

}
