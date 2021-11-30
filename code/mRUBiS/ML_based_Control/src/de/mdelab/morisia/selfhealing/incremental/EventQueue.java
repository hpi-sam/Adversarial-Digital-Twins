package de.mdelab.morisia.selfhealing.incremental;

import java.util.LinkedList;
import java.util.Queue;

import org.eclipse.emf.common.notify.Notification;

public class EventQueue {
	
	// all change events
	public static Queue<Notification> EVENTS = new LinkedList<>();

}
