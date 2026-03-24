import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebar: SidebarsConfig = {
  apisidebar: [
    {
      type: "doc",
      id: "api-reference/generated/memintel-app-developer-api",
    },
    {
      type: "category",
      label: "Tasks",
      link: {
        type: "doc",
        id: "api-reference/generated/tasks",
      },
      items: [
        {
          type: "doc",
          id: "api-reference/generated/create-task",
          label: "Create a task from natural language intent",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-reference/generated/list-tasks",
          label: "List tasks",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-reference/generated/get-task",
          label: "Get task details",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-reference/generated/update-task",
          label: "Update a task",
          className: "api-method patch",
        },
        {
          type: "doc",
          id: "api-reference/generated/delete-task",
          label: "Delete a task",
          className: "api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "Execution",
      link: {
        type: "doc",
        id: "api-reference/generated/execution",
      },
      items: [
        {
          type: "doc",
          id: "api-reference/generated/evaluate-full",
          label: "Execute concept + condition + action pipeline",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Conditions",
      link: {
        type: "doc",
        id: "api-reference/generated/conditions",
      },
      items: [
        {
          type: "doc",
          id: "api-reference/generated/get-condition",
          label: "Get a condition definition",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-reference/generated/explain-condition",
          label: "Explain condition logic and parameters",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-reference/generated/calibrate-condition",
          label: "Generate calibration recommendation for a condition",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-reference/generated/apply-calibration",
          label: "Apply a calibration recommendation as a new condition version",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Decisions",
      link: {
        type: "doc",
        id: "api-reference/generated/decisions",
      },
      items: [
        {
          type: "doc",
          id: "api-reference/generated/explain-decision",
          label: "Explain a specific decision result for an entity",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Feedback",
      link: {
        type: "doc",
        id: "api-reference/generated/feedback",
      },
      items: [
        {
          type: "doc",
          id: "api-reference/generated/submit-feedback",
          label: "Submit feedback on a decision",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Schemas",
      items: [
        {
          type: "doc",
          id: "api-reference/generated/schemas/deliveryconfig",
          label: "DeliveryConfig",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/constraintsconfig",
          label: "ConstraintsConfig",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/task",
          label: "Task",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/tasklist",
          label: "TaskList",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/taskupdaterequest",
          label: "TaskUpdateRequest",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/fullpipelineresult",
          label: "FullPipelineResult",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/conceptresult",
          label: "ConceptResult",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/decisionresult",
          label: "DecisionResult",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/actiontriggered",
          label: "ActionTriggered",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/conditiondefinition",
          label: "ConditionDefinition",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/strategydefinition",
          label: "StrategyDefinition",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/conditionexplanation",
          label: "ConditionExplanation",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/decisionexplanation",
          label: "DecisionExplanation",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/calibraterequest",
          label: "CalibrateRequest",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/calibrationresult",
          label: "CalibrationResult",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/applycalibrationrequest",
          label: "ApplyCalibrationRequest",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/applycalibrationresult",
          label: "ApplyCalibrationResult",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/feedbackrequest",
          label: "FeedbackRequest",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/dryrunresult",
          label: "DryRunResult",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/validationresult",
          label: "ValidationResult",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/validationerror",
          label: "ValidationError",
          className: "schema",
        },
        {
          type: "doc",
          id: "api-reference/generated/schemas/errorresponse",
          label: "ErrorResponse",
          className: "schema",
        },
      ],
    },
  ],
};

export default sidebar.apisidebar;
