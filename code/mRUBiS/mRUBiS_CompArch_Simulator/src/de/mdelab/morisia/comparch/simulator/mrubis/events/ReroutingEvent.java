package de.mdelab.morisia.comparch.simulator.mrubis.events;

import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.RequiredInterface;

/**
 * Event notifying about a re-routing, that is, the removal and addition of a
 * connector in the model to wire a required interface to another provided
 * interface.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class ReroutingEvent {

	private RequiredInterface requiredInterface;
	private Connector oldConnector;
	private Connector newConnector;
	private long timestamp;

	ReroutingEvent(RequiredInterface requiredInterface, Connector oldConnector,
			Connector newConnector) {
		super();
		this.timestamp = System.currentTimeMillis();
		this.requiredInterface = requiredInterface;
		this.oldConnector = oldConnector;
		this.newConnector = newConnector;
	}

	/**
	 * 
	 * @return the required interface that is re-routed.
	 */
	public RequiredInterface getRequiredInterface() {
		return this.requiredInterface;
	}

	/**
	 * 
	 * @return the old connector of the required interface.
	 */
	public Connector getOldConnector() {
		return this.oldConnector;
	}

	/**
	 * 
	 * @return the new connector of the required interface.
	 */
	public Connector getNewConnector() {
		return this.newConnector;
	}

	/**
	 * 
	 * @return the timestamp when the re-routing has been observed.
	 */
	public long getTimestamp() {
		return this.timestamp;
	}

}
