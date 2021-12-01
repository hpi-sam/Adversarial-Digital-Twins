package de.mdelab.morisia.comparch.simulator.mrubis.events;

import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;

/**
 * Event notifying about a change of the life cycle state of a component in the
 * model.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class LifecycleChangeEvent {

	private ComponentLifeCycle oldState;
	private ComponentLifeCycle newState;
	private Component component;
	private String componentId;
	private String componentName;
	private long timestamp;

	public LifecycleChangeEvent(Component component, String componentId,
			String componentName, ComponentLifeCycle oldState,
			ComponentLifeCycle newState) {
		this.timestamp = System.currentTimeMillis();
		this.oldState = oldState;
		this.newState = newState;
		this.component = component;
		this.componentId = componentId;
		this.componentName = componentName;
	}

	/**
	 * @return the old life cycle state of the component.
	 */
	public ComponentLifeCycle getOldState() {
		return this.oldState;
	}

	/**
	 * @return the new life cycle state of the component.
	 */
	public ComponentLifeCycle getNewState() {
		return this.newState;
	}

	/**
	 * @return the component
	 */
	public Component getComponent() {
		return this.component;
	}

	/**
	 * @return the Id of the component
	 */
	public String getComponentId() {
		return this.componentId;
	}

	/**
	 * @return the name of the component
	 */
	public String getComponentName() {
		return this.componentName;
	}

	/**
	 * @return the timestamp when the life cycle change has been observed.
	 */
	public long getTimestamp() {
		return this.timestamp;
	}

}
