package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;

/**
 * Injects a {@link IssueType#CF1}.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class CF1_Injector implements Injector<Component> {

	private final static Logger LOGGER = Logger.getLogger(CF1_Injector.class
			.getName());

	/**
	 * Constructor.
	 */
	public CF1_Injector() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "CF1 Injector";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Sets the life cycle state of a STARTED component to NOT_SUPPORTED.";
	}

	/**
	 * {@inheritDoc}
	 */
	public IssueType getIssueType() {
		return IssueType.CF1;
	}

	/**
	 * Injects a CF1 into the given component.
	 * 
	 * {@inheritDoc}
	 */
	@Override
	public boolean inject(Component component) {
		if (component.getState() == ComponentLifeCycle.STARTED) {
			component.setState(ComponentLifeCycle.NOT_SUPPORTED);
			

			LOGGER.info(" -- Set the life cycle state of component \""
					+ component.getUid() + "\" from tenant \""
					+ component.getTenant().getName() + "\" to NOT_SUPPORTED");

			return true;
		} else {
			LOGGER.info(" -- Cannot set the life cycle state of component \""
					+ component.getUid() + "\" from tenant \""
					+ component.getTenant().getName() + "\" to NOT_SUPPORTED");
			return false;
		}
	}

}
