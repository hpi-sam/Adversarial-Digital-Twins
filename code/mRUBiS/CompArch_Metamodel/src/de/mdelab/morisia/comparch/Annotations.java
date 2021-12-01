/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Annotations</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Annotations#getArchitecture <em>Architecture</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Annotations#getIssues <em>Issues</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Annotations#getRules <em>Rules</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getAnnotations()
 * @model
 * @generated
 */
public interface Annotations extends EObject {
	/**
	 * Returns the value of the '<em><b>Architecture</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Architecture#getAnnotations <em>Annotations</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Architecture</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Architecture</em>' container reference.
	 * @see #setArchitecture(Architecture)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getAnnotations_Architecture()
	 * @see de.mdelab.morisia.comparch.Architecture#getAnnotations
	 * @model opposite="annotations" required="true" transient="false"
	 * @generated
	 */
	Architecture getArchitecture();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Annotations#getArchitecture <em>Architecture</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Architecture</em>' container reference.
	 * @see #getArchitecture()
	 * @generated
	 */
	void setArchitecture(Architecture value);

	/**
	 * Returns the value of the '<em><b>Issues</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Issue}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Issue#getAnnotations <em>Annotations</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Issues</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Issues</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getAnnotations_Issues()
	 * @see de.mdelab.morisia.comparch.Issue#getAnnotations
	 * @model opposite="annotations" containment="true"
	 * @generated
	 */
	EList<Issue> getIssues();

	/**
	 * Returns the value of the '<em><b>Rules</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Rule}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Rule#getAnnotations <em>Annotations</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Rules</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Rules</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getAnnotations_Rules()
	 * @see de.mdelab.morisia.comparch.Rule#getAnnotations
	 * @model opposite="annotations" containment="true"
	 * @generated
	 */
	EList<Rule> getRules();

} // Annotations
