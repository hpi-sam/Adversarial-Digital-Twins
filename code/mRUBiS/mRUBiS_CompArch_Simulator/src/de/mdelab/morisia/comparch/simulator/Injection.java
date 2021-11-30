package de.mdelab.morisia.comparch.simulator;

import de.mdelab.morisia.comparch.ArchitecturalElement;

/**
 * Specification of an injection of an issue into the model, which is going to
 * be executed by the simulator.
 * 
 * @author thomas vogel
 * @version 0.30
 *
 * @param <T>
 *            the type of model element to which the issue is injected.
 */
public class Injection<T extends ArchitecturalElement> {

	private T target;
	private IssueType issueType;
	private boolean success;

	/**
	 * Creates an injection specification.
	 * 
	 * @param issueType
	 *            the type of issue to be injected.
	 * @param target
	 *            the model element to which an issue of the given type is going
	 *            to be injected.
	 */
	public Injection(IssueType issueType, T target) {
		super();
		this.issueType = issueType;
		this.target = target;
	}

	/**
	 * @return the model element to which an issue of the given type is (going
	 *         to be) injected.
	 */
	public T getTarget() {
		return this.target;
	}

	/**
	 * @return the type of the issue that is (going to be) injected.
	 */
	public IssueType getIssueType() {
		return this.issueType;
	}

	/**
	 * @return after the injection, <code>true</code> if the simulator
	 *         successfully injected the issue, otherwise <code>false</code>.
	 */
	public boolean isSuccess() {
		return this.success;
	}

	/**
	 * Sets the success of the injection. This method is used by the simulator
	 * to notify about the success of an injection.
	 * 
	 * @param success
	 *            <code>true</code> if the simulator successfully injected the
	 *            issue, otherwise <code>false</code>.
	 */
	public void setSuccess(boolean success) {
		this.success = success;
	}

}
