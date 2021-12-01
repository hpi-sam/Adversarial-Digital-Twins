/**
 */
package de.mdelab.morisia.comparch;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Failure</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Failure#getName <em>Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Failure#getMessage <em>Message</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Failure#getInterface <em>Interface</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Failure#getMethod <em>Method</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getFailure()
 * @model
 * @generated
 */
public interface Failure extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getFailure_Name()
	 * @model required="true"
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Failure#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Message</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Message</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Message</em>' attribute.
	 * @see #setMessage(String)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getFailure_Message()
	 * @model
	 * @generated
	 */
	String getMessage();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Failure#getMessage <em>Message</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Message</em>' attribute.
	 * @see #getMessage()
	 * @generated
	 */
	void setMessage(String value);

	/**
	 * Returns the value of the '<em><b>Interface</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ProvidedInterface#getFailures <em>Failures</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Interface</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Interface</em>' container reference.
	 * @see #setInterface(ProvidedInterface)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getFailure_Interface()
	 * @see de.mdelab.morisia.comparch.ProvidedInterface#getFailures
	 * @model opposite="failures" required="true" transient="false"
	 * @generated
	 */
	ProvidedInterface getInterface();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Failure#getInterface <em>Interface</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Interface</em>' container reference.
	 * @see #getInterface()
	 * @generated
	 */
	void setInterface(ProvidedInterface value);

	/**
	 * Returns the value of the '<em><b>Method</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Method</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Method</em>' reference.
	 * @see #setMethod(MethodSpecification)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getFailure_Method()
	 * @model required="true"
	 * @generated
	 */
	MethodSpecification getMethod();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Failure#getMethod <em>Method</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Method</em>' reference.
	 * @see #getMethod()
	 * @generated
	 */
	void setMethod(MethodSpecification value);

} // Failure
