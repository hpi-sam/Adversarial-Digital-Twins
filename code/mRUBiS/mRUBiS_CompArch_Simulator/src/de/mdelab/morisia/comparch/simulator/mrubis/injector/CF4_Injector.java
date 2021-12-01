package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import java.util.logging.Logger;

import org.eclipse.emf.ecore.util.EcoreUtil;

import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;

/**
 * Injects a {@link IssueType#CF4}.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class CF4_Injector implements Injector<Connector> {

	private final static Logger LOGGER = Logger.getLogger(CF4_Injector.class
			.getName());

	/**
	 * Constructor.
	 */
	public CF4_Injector() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "CF4 Injector";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Removes a connector from a STARTED component (the one indicated by the parameter of the inject method) "
				+ "to another STARTED component.";
	}

	/**
	 * {@inheritDoc}
	 */
	public IssueType getIssueType() {
		return IssueType.CF4;
	}

	/**
	 * Injects a CF5 by removing the given connector.
	 * 
	 * {@inheritDoc}
	 */
	@Override
	public boolean inject(Connector connector) {
		String connectorId = connector.getUid();
		// check if connector is properly connected, that is, source and
		// target exist and they belong to started components.
		if (connector.getSource() != null
				&& connector.getSource().getComponent() != null
				&& connector.getSource().getComponent().getState() == ComponentLifeCycle.STARTED
				&& connector.getTarget() != null
				&& connector.getTarget().getComponent() != null
				&& connector.getTarget().getComponent().getState() == ComponentLifeCycle.STARTED) {
			// remove connector
			Component sourceComponent = connector.getSource().getComponent();
			Component targetComponent = connector.getTarget().getComponent();
			EcoreUtil.delete(connector, true);
			LOGGER.info(" -- Completely deleted connector \"" + connectorId
					+ "\" from component \"" + sourceComponent.getUid()
					+ "\" to component \"" + targetComponent.getUid()
					+ "\" in tenant \"" + sourceComponent.getTenant().getName()
					+ "\".");
			return true;
		} else {
			// connector is not properly connected
			LOGGER.info(" -- Cannot delete connector \"" + connectorId
					+ "\" since the connector is not properly connected.");
			return false;
		}
	}

}
