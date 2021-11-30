package de.mdelab.morisia.comparch.simulator.mrubis.events;

import java.util.LinkedList;
import java.util.Queue;

/**
 * Queue of {@link LifecycleChangeEvent}s and {@link ReroutingEvent}s that can
 * be used for analyzing an adaptation performed by a feedback loop that handles
 * the injected issues.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class ChangeEventQueue {

	/**
	 * Singleton instance of the queue.
	 */
	public final static ChangeEventQueue INSTANCE = new ChangeEventQueue();

	private Queue<LifecycleChangeEvent> lifecycleEvents;
	private Queue<ReroutingEvent> reroutingEvents;

	/**
	 * Hidden constructor.
	 */
	private ChangeEventQueue() {
		this.lifecycleEvents = new LinkedList<LifecycleChangeEvent>();
		this.reroutingEvents = new LinkedList<ReroutingEvent>();
	}

	/**
	 * Adds a {@code LifecycleChangeEvent} to the queue.
	 * 
	 * @param event
	 *            the event to be added.
	 */
	public void addEvent(LifecycleChangeEvent event) {
		this.lifecycleEvents.add(event);
	}

	/**
	 * Returns the queue of {@code LifecycleChangeEvent}s.
	 * 
	 * @return the queue of {@code LifecycleChangeEvent}s.
	 */
	public Queue<LifecycleChangeEvent> getLifecycleEvents() {
		return this.lifecycleEvents;
	}

	/**
	 * Adds a {@code ReroutingEvent} to the queue.
	 * 
	 * @param event
	 *            the event to be added.
	 */
	public void addEvent(ReroutingEvent event) {
		this.reroutingEvents.add(event);
	}

	/**
	 * Returns the queue of {@code ReroutingEvent}s.
	 * 
	 * @return the queue of {@code ReroutingEvent}s.
	 */
	public Queue<ReroutingEvent> getReroutingEvents() {
		return this.reroutingEvents;
	}

	public void clearEvents() {
		this.lifecycleEvents.clear();
		this.reroutingEvents.clear();
	}

}
