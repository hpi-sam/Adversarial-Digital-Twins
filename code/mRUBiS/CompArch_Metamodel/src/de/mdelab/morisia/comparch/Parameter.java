/**
 */
package de.mdelab.morisia.comparch;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Parameter</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Parameter#getValue <em>Value</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Parameter#isLazyAdaptation <em>Lazy Adaptation</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Parameter#getComponent <em>Component</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Parameter#getType <em>Type</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getParameter()
 * @model
 * @generated
 */
public interface Parameter extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Value</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Value</em>' attribute.
	 * @see #setValue(String)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getParameter_Value()
	 * @model required="true"
	 * @generated
	 */
	String getValue();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Parameter#getValue <em>Value</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Value</em>' attribute.
	 * @see #getValue()
	 * @generated
	 */
	void setValue(String value);

	/**
	 * Returns the value of the '<em><b>Lazy Adaptation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Lazy Adaptation</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Lazy Adaptation</em>' attribute.
	 * @see #setLazyAdaptation(boolean)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getParameter_LazyAdaptation()
	 * @model required="true"
	 * @generated
	 */
	boolean isLazyAdaptation();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Parameter#isLazyAdaptation <em>Lazy Adaptation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Lazy Adaptation</em>' attribute.
	 * @see #isLazyAdaptation()
	 * @generated
	 */
	void setLazyAdaptation(boolean value);

	/**
	 * Returns the value of the '<em><b>Component</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Component#getParameters <em>Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Component</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Component</em>' container reference.
	 * @see #setComponent(Component)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getParameter_Component()
	 * @see de.mdelab.morisia.comparch.Component#getParameters
	 * @model opposite="parameters" required="true" transient="false"
	 * @generated
	 */
	Component getComponent();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Parameter#getComponent <em>Component</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Component</em>' container reference.
	 * @see #getComponent()
	 * @generated
	 */
	void setComponent(Component value);

	/**
	 * Returns the value of the '<em><b>Type</b></em>' reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ParameterType#getInstances <em>Instances</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Type</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Type</em>' reference.
	 * @see #setType(ParameterType)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getParameter_Type()
	 * @see de.mdelab.morisia.comparch.ParameterType#getInstances
	 * @model opposite="instances" required="true"
	 * @generated
	 */
	ParameterType getType();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Parameter#getType <em>Type</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Type</em>' reference.
	 * @see #getType()
	 * @generated
	 */
	void setType(ParameterType value);

} // Parameter
