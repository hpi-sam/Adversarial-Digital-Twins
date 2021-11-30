/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Interface Type</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.InterfaceType#getFullyQualifiedName <em>Fully Qualified Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.InterfaceType#getArchitecture <em>Architecture</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.InterfaceType#getRequiringComponentTypes <em>Requiring Component Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.InterfaceType#getProvidingComponentTypes <em>Providing Component Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.InterfaceType#getMethodSpecification <em>Method Specification</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.InterfaceType#getInstances <em>Instances</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterfaceType()
 * @model
 * @generated
 */
public interface InterfaceType extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Fully Qualified Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Fully Qualified Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Fully Qualified Name</em>' attribute.
	 * @see #setFullyQualifiedName(String)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterfaceType_FullyQualifiedName()
	 * @model
	 * @generated
	 */
	String getFullyQualifiedName();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.InterfaceType#getFullyQualifiedName <em>Fully Qualified Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Fully Qualified Name</em>' attribute.
	 * @see #getFullyQualifiedName()
	 * @generated
	 */
	void setFullyQualifiedName(String value);

	/**
	 * Returns the value of the '<em><b>Architecture</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Architecture#getInterfaceTypes <em>Interface Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Architecture</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Architecture</em>' container reference.
	 * @see #setArchitecture(Architecture)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterfaceType_Architecture()
	 * @see de.mdelab.morisia.comparch.Architecture#getInterfaceTypes
	 * @model opposite="interfaceTypes" required="true" transient="false"
	 * @generated
	 */
	Architecture getArchitecture();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.InterfaceType#getArchitecture <em>Architecture</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Architecture</em>' container reference.
	 * @see #getArchitecture()
	 * @generated
	 */
	void setArchitecture(Architecture value);

	/**
	 * Returns the value of the '<em><b>Requiring Component Types</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.ComponentType}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ComponentType#getRequiredInterfaceTypes <em>Required Interface Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Requiring Component Types</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Requiring Component Types</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterfaceType_RequiringComponentTypes()
	 * @see de.mdelab.morisia.comparch.ComponentType#getRequiredInterfaceTypes
	 * @model opposite="requiredInterfaceTypes"
	 * @generated
	 */
	EList<ComponentType> getRequiringComponentTypes();

	/**
	 * Returns the value of the '<em><b>Providing Component Types</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.ComponentType}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ComponentType#getProvidedInterfaceTypes <em>Provided Interface Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Providing Component Types</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Providing Component Types</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterfaceType_ProvidingComponentTypes()
	 * @see de.mdelab.morisia.comparch.ComponentType#getProvidedInterfaceTypes
	 * @model opposite="providedInterfaceTypes"
	 * @generated
	 */
	EList<ComponentType> getProvidingComponentTypes();

	/**
	 * Returns the value of the '<em><b>Method Specification</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.MethodSpecification}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.MethodSpecification#getInterfaceType <em>Interface Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Method Specification</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Method Specification</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterfaceType_MethodSpecification()
	 * @see de.mdelab.morisia.comparch.MethodSpecification#getInterfaceType
	 * @model opposite="interfaceType" containment="true" required="true"
	 * @generated
	 */
	EList<MethodSpecification> getMethodSpecification();

	/**
	 * Returns the value of the '<em><b>Instances</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Interface}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Interface#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Instances</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Instances</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getInterfaceType_Instances()
	 * @see de.mdelab.morisia.comparch.Interface#getType
	 * @model opposite="type"
	 * @generated
	 */
	EList<Interface> getInstances();

} // InterfaceType
