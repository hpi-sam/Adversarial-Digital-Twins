package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;

/**
 * Injects a {@link IssueType#CF3}.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class CF3_Injector implements Injector<Component> {

	private final static Logger LOGGER = Logger.getLogger(CF3_Injector.class
			.getName());

	/**
	 * Constructor.
	 */
	public CF3_Injector() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "CF3 Injector";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Removes a STARTED component from the model.";
	}

	/**
	 * {@inheritDoc}
	 */
	public IssueType getIssueType() {
		return IssueType.CF3;
	}

	/**
	 * Injects a CF3 by removing the given component.
	 * 
	 * {@inheritDoc}
	 */
	@Override
	public boolean inject(Component component) {
		String componentUid = component.getUid();
		ComponentType componentType = component.getType();
		Tenant tenant = component.getTenant();
		if (component.getState() == ComponentLifeCycle.STARTED) {
			// EcoreUtil.delete(component, true);
			component.setState(ComponentLifeCycle.REMOVED);
			LOGGER.info(" -- Removed component \"" + componentUid
					+ "\" of type \"" + componentType.getName()
					+ "\" from tenant \"" + tenant.getName() + "\".");
			return true;
		} else {
			// Component is not started
			LOGGER.info(" -- Cannot remove non-running component \""
					+ componentUid + "\" of type \"" + componentType.getName()
					+ "\" from tenant \"" + tenant.getName() + "\".");
			return false;
		}
	}
}
