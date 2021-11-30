package de.mdelab.morisia.comparch.simulator;

import de.mdelab.morisia.comparch.ArchitecturalElement;

/**
 * Injects an issue into the model.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public interface Injector<T extends ArchitecturalElement> {

	/**
	 * @return the name of the injector.
	 */
	public String getName();

	/**
	 * @return the description of the injector.
	 */
	public String getDescription();

	/**
	 * @return the type of issues injected by this injector.
	 */
	public IssueType getIssueType();

	/**
	 * Injects an issue into the model, particularly, to the given target.
	 * 
	 * @param target
	 *            The target to which the issue should be injected.
	 * @return <code>true</code> if the issue was successfully injected,
	 *         otherwise <code>false</code>.
	 */
	public boolean inject(T target);


}
