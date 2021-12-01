package de.mdelab.morisia.selfhealing.rules;

import de.mdelab.morisia.comparch.CF;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Rule;

public class ExecuteCleanUp {

	public static void removeMarkers(CF cf, Rule rule) {
		// rm rule
		//rule.setAnnotations(null);
		// rule.setHandles(null);
		// rm cf
		// cf.setAnnotations(null);
		cf.setAffectedComponent(null);
	}

	public static void removeFailures(ProvidedInterface providedInterface) {
		providedInterface.getFailures().clear();
	}

}
