/**
 */
package de.mdelab.morisia.comparch;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Replace Component</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.ReplaceComponent#getAlternativeComponentType <em>Alternative Component Type</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplaceComponent()
 * @model
 * @generated
 */
public interface ReplaceComponent extends Rule {

	/**
	 * Returns the value of the '<em><b>Alternative Component Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Alternative Component Type</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Alternative Component Type</em>' reference.
	 * @see #setAlternativeComponentType(ComponentType)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplaceComponent_AlternativeComponentType()
	 * @model
	 * @generated
	 */
	ComponentType getAlternativeComponentType();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ReplaceComponent#getAlternativeComponentType <em>Alternative Component Type</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Alternative Component Type</em>' reference.
	 * @see #getAlternativeComponentType()
	 * @generated
	 */
	void setAlternativeComponentType(ComponentType value);
} // ReplaceComponent
