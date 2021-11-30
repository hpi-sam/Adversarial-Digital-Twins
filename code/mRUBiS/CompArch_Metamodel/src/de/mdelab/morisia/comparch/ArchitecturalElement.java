/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Architectural Element</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.ArchitecturalElement#getUid <em>Uid</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ArchitecturalElement#getCriticality <em>Criticality</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ArchitecturalElement#getImportance <em>Importance</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ArchitecturalElement#getADT <em>ADT</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ArchitecturalElement#getEfficiency <em>Efficiency</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ArchitecturalElement#getPerformance <em>Performance</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ArchitecturalElement#getRequest <em>Request</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecturalElement()
 * @model abstract="true"
 * @generated
 */
public interface ArchitecturalElement extends EObject {
	/**
	 * Returns the value of the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Uid</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Uid</em>' attribute.
	 * @see #setUid(String)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecturalElement_Uid()
	 * @model required="true"
	 * @generated
	 */
	String getUid();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getUid <em>Uid</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Uid</em>' attribute.
	 * @see #getUid()
	 * @generated
	 */
	void setUid(String value);

	/**
	 * Returns the value of the '<em><b>Criticality</b></em>' attribute.
	 * The default value is <code>"0.0"</code>.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Criticality</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Criticality</em>' attribute.
	 * @see #setCriticality(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecturalElement_Criticality()
	 * @model default="0.0"
	 * @generated
	 */
	double getCriticality();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getCriticality <em>Criticality</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Criticality</em>' attribute.
	 * @see #getCriticality()
	 * @generated
	 */
	void setCriticality(double value);

	/**
	 * Returns the value of the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Importance</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Importance</em>' attribute.
	 * @see #setImportance(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecturalElement_Importance()
	 * @model
	 * @generated
	 */
	double getImportance();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getImportance <em>Importance</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Importance</em>' attribute.
	 * @see #getImportance()
	 * @generated
	 */
	void setImportance(double value);

	/**
	 * Returns the value of the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>ADT</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>ADT</em>' attribute.
	 * @see #setADT(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecturalElement_ADT()
	 * @model
	 * @generated
	 */
	double getADT();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getADT <em>ADT</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>ADT</em>' attribute.
	 * @see #getADT()
	 * @generated
	 */
	void setADT(double value);

	/**
	 * Returns the value of the '<em><b>Efficiency</b></em>' attribute.
	 * The default value is <code>"0.0"</code>.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Efficiency</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Efficiency</em>' attribute.
	 * @see #setEfficiency(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecturalElement_Efficiency()
	 * @model default="0.0"
	 * @generated
	 */
	double getEfficiency();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getEfficiency <em>Efficiency</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Efficiency</em>' attribute.
	 * @see #getEfficiency()
	 * @generated
	 */
	void setEfficiency(double value);

	/**
	 * Returns the value of the '<em><b>Performance</b></em>' attribute.
	 * The default value is <code>"0.0"</code>.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Performance</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Performance</em>' attribute.
	 * @see #setPerformance(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecturalElement_Performance()
	 * @model default="0.0"
	 * @generated
	 */
	double getPerformance();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getPerformance <em>Performance</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Performance</em>' attribute.
	 * @see #getPerformance()
	 * @generated
	 */
	void setPerformance(double value);

	/**
	 * Returns the value of the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Request</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Request</em>' attribute.
	 * @see #setRequest(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecturalElement_Request()
	 * @model
	 * @generated
	 */
	double getRequest();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getRequest <em>Request</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Request</em>' attribute.
	 * @see #getRequest()
	 * @generated
	 */
	void setRequest(double value);

} // ArchitecturalElement
