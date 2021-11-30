/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Rule</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Rule#getCosts <em>Costs</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Rule#getAnnotations <em>Annotations</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Rule#getHandles <em>Handles</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Rule#getUtilityIncrease <em>Utility Increase</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getRule()
 * @model abstract="true"
 * @generated
 */
public interface Rule extends EObject {
	/**
	 * Returns the value of the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Costs</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Costs</em>' attribute.
	 * @see #setCosts(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getRule_Costs()
	 * @model
	 * @generated
	 */
	double getCosts();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Rule#getCosts <em>Costs</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Costs</em>' attribute.
	 * @see #getCosts()
	 * @generated
	 */
	void setCosts(double value);

	/**
	 * Returns the value of the '<em><b>Annotations</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Annotations#getRules <em>Rules</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Annotations</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Annotations</em>' container reference.
	 * @see #setAnnotations(Annotations)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getRule_Annotations()
	 * @see de.mdelab.morisia.comparch.Annotations#getRules
	 * @model opposite="rules" required="true" transient="false"
	 * @generated
	 */
	Annotations getAnnotations();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Rule#getAnnotations <em>Annotations</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Annotations</em>' container reference.
	 * @see #getAnnotations()
	 * @generated
	 */
	void setAnnotations(Annotations value);

	/**
	 * Returns the value of the '<em><b>Handles</b></em>' reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Issue#getHandledBy <em>Handled By</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Handles</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Handles</em>' reference.
	 * @see #setHandles(Issue)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getRule_Handles()
	 * @see de.mdelab.morisia.comparch.Issue#getHandledBy
	 * @model opposite="handledBy" required="true"
	 * @generated
	 */
	Issue getHandles();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Rule#getHandles <em>Handles</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Handles</em>' reference.
	 * @see #getHandles()
	 * @generated
	 */
	void setHandles(Issue value);

	/**
	 * Returns the value of the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Utility Increase</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Utility Increase</em>' attribute.
	 * @see #setUtilityIncrease(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getRule_UtilityIncrease()
	 * @model
	 * @generated
	 */
	double getUtilityIncrease();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Rule#getUtilityIncrease <em>Utility Increase</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Utility Increase</em>' attribute.
	 * @see #getUtilityIncrease()
	 * @generated
	 */
	void setUtilityIncrease(double value);

} // Rule
