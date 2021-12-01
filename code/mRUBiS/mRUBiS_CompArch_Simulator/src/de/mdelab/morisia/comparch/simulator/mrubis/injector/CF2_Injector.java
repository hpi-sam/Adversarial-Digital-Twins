package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import java.util.List;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.Failure;
import de.mdelab.morisia.comparch.MethodSpecification;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;
import de.mdelab.morisia.comparch.simulator.impl.SimulatorUtil;

/**
 * Injects a {@link IssueType#CF2}.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class CF2_Injector implements Injector<ProvidedInterface> {

	private final static Logger LOGGER = Logger.getLogger(CF2_Injector.class
			.getName());

	private int failureThreshold;

	/**
	 * Constructor.
	 * 
	 * @param failureThreshold
	 *            number of failure elements that cause a CF2. This injector
	 *            will add at least so many failure elements.
	 */
	public CF2_Injector(int failureThreshold) {
		this.failureThreshold = failureThreshold;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "CF2 (>= #" + this.failureThreshold + " Failures) Injector";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Adds at least " + this.failureThreshold
				+ " failures to a provided interface of a STARTED component.";
	}

	/**
	 * {@inheritDoc}
	 */
	public IssueType getIssueType() {
		return IssueType.CF2;
	}

	/**
	 * Injects a CF2 into the given provided interface of a STARTED component.
	 * 
	 * {@inheritDoc}
	 */
	@Override
	public boolean inject(ProvidedInterface providedInterface) {
		Component component = providedInterface.getComponent();

		if (component.getState() == ComponentLifeCycle.STARTED) {

//			// identify whether the provided interface is used by at least
//			// one STARTED component.
//			boolean isUsed = false;
//			for (Connector conn : providedInterface.getConnectors()) {
//				RequiredInterface ri = conn.getSource();
//				if (ri != null) {
//					Component c = ri.getComponent();
//					if (c != null && c.getState() == ComponentLifeCycle.STARTED) {
//						isUsed = true;
//						break;
//					}
//				}
//			}
			
//			if (isUsed) {
				int numberOfFailures = this.failureThreshold
						+ SimulatorUtil.randInt(1, this.failureThreshold);
				for (int i = 0; i < numberOfFailures; i++) {
					this.addFailure(providedInterface,
							SimulatorUtil.generateUUID() + "_Failure",
							"java.lang.RuntimeException", "Failure #" + i);
				}

				LOGGER.info(" -- Attached " + numberOfFailures
						+ " failures to the interface \""
						+ providedInterface.getUid()
						+ "\" provided by component \"" + component.getUid()
						+ "\" of tenant \"" + component.getTenant().getName()
						+ "\".");
				return true;
//			} else {
//				// The provided interface is not used
//				LOGGER.info(" -- Cannot attach any failures to the unused interface \""
//						+ providedInterface.getUid()
//						+ "\" provided by component \""
//						+ component.getUid()
//						+ "\" of tenant \""
//						+ component.getTenant().getName()
//						+ "\".");
//				return false;
//			}
		} else {
			// component is not started
			LOGGER.info(" -- Cannot attach any failures to the interface \""
					+ providedInterface.getUid()
					+ "\" provided by the non-running component \""
					+ component.getUid() + "\" of tenant \""
					+ component.getTenant().getName() + "\".");
			return false;
		}

	}

	/**
	 * Creates and adds a failure element to the given provided interface.
	 * 
	 * @param eProInterface
	 *            the given provided interface
	 * @param uid
	 *            the uid of the failure
	 * @param name
	 *            the name of the failure
	 * @param message
	 *            the message of the failure
	 */
	private void addFailure(ProvidedInterface eProInterface, String uid,
			String name, String message) {
		Failure eFailure = ComparchFactory.eINSTANCE.createFailure();
		eFailure.setUid(uid);
		eFailure.setName(name);
		eFailure.setMessage(message);

		List<MethodSpecification> eMethods = eProInterface.getType()
				.getMethodSpecification();
		int numberOfMethods = eMethods.size();
		if (numberOfMethods > 1) {
			int no = SimulatorUtil.randInt(0, numberOfMethods - 1);
			eFailure.setMethod(eMethods.get(no));
		} else {
			eFailure.setMethod(eMethods.get(0));
		}
		eProInterface.getFailures().add(eFailure);
	}

}
