package de.mdelab.morisia.selfhealing.incremental;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.util.EContentAdapter;

public class EventListener extends EContentAdapter {

	public EventListener() {
		super();
	}

	@Override
	public void notifyChanged(Notification notification) {
		super.notifyChanged(notification);
		EventQueue.EVENTS.add(notification);
	}

	@Override
	public boolean isAdapterForType(Object type) {
		return type == EventListener.class;
	}

}
