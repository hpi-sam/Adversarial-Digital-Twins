package de.mdelab.morisia.comparch.simulator.mrubis.events;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.util.EContentAdapter;

import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.RequiredInterface;

/**
 * Collects change events of the model.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class ChangeEventObserver extends EContentAdapter {

	/**
	 * Constructor.
	 */
	public ChangeEventObserver() {
		super();
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Filters relevant notifications and adds corresponding events to the
	 * queues.
	 */
	@Override
	public void notifyChanged(Notification notification) {
		super.notifyChanged(notification);

		Object notifier = notification.getNotifier();
		Object feature = notification.getFeature();

		// check changes in the attribute Component.state
		if (feature == ComparchPackage.Literals.COMPONENT__STATE) {
			assert notifier instanceof Component;
			Component component = (Component) notifier;
			ComponentLifeCycle oldState = (ComponentLifeCycle) notification
					.getOldValue();
			ComponentLifeCycle newState = (ComponentLifeCycle) notification
					.getNewValue();
			ChangeEventQueue.INSTANCE.addEvent(new LifecycleChangeEvent(
					component, component.getUid(), component.getType()
							.getName(), oldState, newState));
		} else
		// check changes to connectors
		if (feature == ComparchPackage.Literals.REQUIRED_INTERFACE__CONNECTOR) {
			assert notifier instanceof RequiredInterface;
			RequiredInterface requiredInterface = (RequiredInterface) notifier;
			Connector oldConnector = (Connector) notification.getOldValue();
			Connector newConnector = (Connector) notification.getNewValue();
			ChangeEventQueue.INSTANCE.addEvent(new ReroutingEvent(
					requiredInterface, oldConnector, newConnector));
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public boolean isAdapterForType(Object type) {
		return type == ChangeEventObserver.class;
	}

}
