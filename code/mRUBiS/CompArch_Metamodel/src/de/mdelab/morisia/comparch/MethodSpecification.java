/**
 */
package de.mdelab.morisia.comparch;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Method Specification</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.MethodSpecification#getSignature <em>Signature</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.MethodSpecification#getInterfaceType <em>Interface Type</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getMethodSpecification()
 * @model
 * @generated
 */
public interface MethodSpecification extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Signature</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Signature</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Signature</em>' attribute.
	 * @see #setSignature(String)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getMethodSpecification_Signature()
	 * @model required="true"
	 * @generated
	 */
	String getSignature();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.MethodSpecification#getSignature <em>Signature</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Signature</em>' attribute.
	 * @see #getSignature()
	 * @generated
	 */
	void setSignature(String value);

	/**
	 * Returns the value of the '<em><b>Interface Type</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.InterfaceType#getMethodSpecification <em>Method Specification</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Interface Type</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Interface Type</em>' container reference.
	 * @see #setInterfaceType(InterfaceType)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getMethodSpecification_InterfaceType()
	 * @see de.mdelab.morisia.comparch.InterfaceType#getMethodSpecification
	 * @model opposite="methodSpecification" required="true" transient="false"
	 * @generated
	 */
	InterfaceType getInterfaceType();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.MethodSpecification#getInterfaceType <em>Interface Type</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Interface Type</em>' container reference.
	 * @see #getInterfaceType()
	 * @generated
	 */
	void setInterfaceType(InterfaceType value);

} // MethodSpecification
