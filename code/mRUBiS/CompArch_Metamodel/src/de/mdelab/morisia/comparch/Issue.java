/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Issue</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Issue#getUtilityDrop <em>Utility Drop</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Issue#getAffectedComponent <em>Affected Component</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Issue#getAnnotations <em>Annotations</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Issue#getHandledBy <em>Handled By</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getIssue()
 * @model abstract="true"
 * @generated
 */
public interface Issue extends EObject {
	/**
	 * Returns the value of the '<em><b>Utility Drop</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Utility Drop</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Utility Drop</em>' attribute.
	 * @see #setUtilityDrop(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getIssue_UtilityDrop()
	 * @model
	 * @generated
	 */
	double getUtilityDrop();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Issue#getUtilityDrop <em>Utility Drop</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Utility Drop</em>' attribute.
	 * @see #getUtilityDrop()
	 * @generated
	 */
	void setUtilityDrop(double value);

	/**
	 * Returns the value of the '<em><b>Affected Component</b></em>' reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Component#getIssues <em>Issues</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Affected Component</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Affected Component</em>' reference.
	 * @see #setAffectedComponent(Component)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getIssue_AffectedComponent()
	 * @see de.mdelab.morisia.comparch.Component#getIssues
	 * @model opposite="issues" required="true"
	 * @generated
	 */
	Component getAffectedComponent();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Issue#getAffectedComponent <em>Affected Component</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Affected Component</em>' reference.
	 * @see #getAffectedComponent()
	 * @generated
	 */
	void setAffectedComponent(Component value);

	/**
	 * Returns the value of the '<em><b>Annotations</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Annotations#getIssues <em>Issues</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Annotations</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Annotations</em>' container reference.
	 * @see #setAnnotations(Annotations)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getIssue_Annotations()
	 * @see de.mdelab.morisia.comparch.Annotations#getIssues
	 * @model opposite="issues" required="true" transient="false"
	 * @generated
	 */
	Annotations getAnnotations();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Issue#getAnnotations <em>Annotations</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Annotations</em>' container reference.
	 * @see #getAnnotations()
	 * @generated
	 */
	void setAnnotations(Annotations value);

	/**
	 * Returns the value of the '<em><b>Handled By</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Rule}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Rule#getHandles <em>Handles</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Handled By</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Handled By</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getIssue_HandledBy()
	 * @see de.mdelab.morisia.comparch.Rule#getHandles
	 * @model opposite="handles"
	 * @generated
	 */
	EList<Rule> getHandledBy();

} // Issue
