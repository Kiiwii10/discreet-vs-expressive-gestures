function createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text) el.textContent = text;
  return el;
}

function fieldInput({ id, label, type = "text", required = false, attrs = {} }) {
  const wrapper = createEl("label", "field");
  wrapper.appendChild(document.createTextNode(label));
  const input = document.createElement("input");
  input.type = type;
  if (id) input.id = id;
  if (required) input.required = true;
  Object.entries(attrs).forEach(([key, value]) => {
    input.setAttribute(key, value);
  });
  wrapper.appendChild(input);
  return wrapper;
}

function optionInput({ type, name, value, text, required = false, id = null }) {
  const label = createEl("label", "option");
  const input = document.createElement("input");
  input.type = type;
  input.name = name;
  input.value = value;
  if (id) input.id = id;
  if (required) input.required = true;
  label.appendChild(input);
  label.appendChild(document.createTextNode(text));
  return label;
}

function buildLikertGrid(namePrefix, rows, t) {
  const grid = createEl("div", "likert-grid");
  const head = createEl("div", "likert-head");
  head.appendChild(createEl("span", "likert-label", t("likert.statement")));
  for (let i = 1; i <= 7; i += 1) {
    head.appendChild(createEl("span", "", String(i)));
  }
  grid.appendChild(head);
  rows.forEach((row) => {
    const rowEl = createEl("div", "likert-row");
    rowEl.appendChild(createEl("div", "likert-label", row.label));
    for (let i = 1; i <= 7; i += 1) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = `${namePrefix}_${row.key}`;
      input.value = String(i);
      if (i === 1) input.required = true;
      label.appendChild(input);
      rowEl.appendChild(label);
    }
    grid.appendChild(rowEl);
  });
  return grid;
}

function buildScaleGrid(name, maxValue) {
  const grid = createEl("div", `scale-grid scale-${maxValue}`);
  for (let i = 1; i <= maxValue; i += 1) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = String(i);
    if (i === 1) input.required = true;
    label.appendChild(input);
    label.appendChild(document.createTextNode(String(i)));
    grid.appendChild(label);
  }
  return grid;
}

export function snapshotForm(container) {
  if (!container) return {};
  const snapshot = { values: {}, checks: {} };
  container.querySelectorAll("input, textarea, select").forEach((input) => {
    const key = input.id || input.name;
    if (!key) return;
    if (input.type === "checkbox") {
      if (!snapshot.checks[key]) snapshot.checks[key] = [];
      if (input.checked) snapshot.checks[key].push(input.value);
      return;
    }
    if (input.type === "radio") {
      if (input.checked) snapshot.values[key] = input.value;
      return;
    }
    snapshot.values[key] = input.value;
  });
  return snapshot;
}

export function restoreForm(container, snapshot) {
  if (!container || !snapshot) return;
  container.querySelectorAll("input, textarea, select").forEach((input) => {
    const key = input.id || input.name;
    if (!key) return;
    if (input.type === "checkbox") {
      const items = snapshot.checks?.[key] || [];
      input.checked = items.includes(input.value);
      return;
    }
    if (input.type === "radio") {
      input.checked = snapshot.values?.[key] === input.value;
      return;
    }
    if (snapshot.values?.[key] != null) {
      input.value = snapshot.values[key];
    }
  });
}

export function resetFormInputs(container) {
  if (!container) return;
  container.querySelectorAll("input").forEach((input) => {
    if (input.type === "checkbox" || input.type === "radio") {
      input.checked = false;
    } else {
      input.value = "";
    }
  });
  container.querySelectorAll("textarea").forEach((input) => {
    input.value = "";
  });
}

export function getCheckedValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

export function getCheckedValues(name) {
  return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map((el) => el.value);
}

export function getInputValue(id) {
  const el = document.getElementById(id);
  if (!el) return "";
  return String(el.value ?? "").trim();
}

export function renderPrestudyForm({ elements, t }) {
  const container = elements.prestudyForm;
  if (!container) return;
  container.innerHTML = "";

  const background = createEl("div", "form-section");
  background.appendChild(createEl("h3", "", t("prestudy.background")));
  const grid = createEl("div", "form-grid");
  grid.appendChild(
    fieldInput({
      id: "age",
      label: t("prestudy.age"),
      type: "number",
      required: true,
      attrs: { min: 13, max: 99 },
    }),
  );
  background.appendChild(grid);

  const gender = createEl("div", "question");
  gender.appendChild(createEl("span", "label", t("prestudy.gender")));
  const genderGrid = createEl("div", "option-grid");
  genderGrid.appendChild(optionInput({ type: "radio", name: "gender", value: "Male", text: t("prestudy.gender.male") }));
  genderGrid.appendChild(optionInput({ type: "radio", name: "gender", value: "Female", text: t("prestudy.gender.female") }));
  genderGrid.appendChild(optionInput({ type: "radio", name: "gender", value: "Other", text: t("prestudy.gender.other") }));
  gender.appendChild(genderGrid);
  background.appendChild(gender);

  const handed = createEl("div", "question");
  handed.appendChild(createEl("span", "label", t("prestudy.handedness")));
  const handGrid = createEl("div", "option-grid");
  handGrid.appendChild(
    optionInput({ type: "radio", name: "handedness", value: "Right", text: t("prestudy.handedness.right"), required: true }),
  );
  handGrid.appendChild(
    optionInput({ type: "radio", name: "handedness", value: "Left", text: t("prestudy.handedness.left"), required: true }),
  );
  handGrid.appendChild(
    optionInput({ type: "radio", name: "handedness", value: "Both", text: t("prestudy.handedness.both"), required: true }),
  );
  handed.appendChild(handGrid);
  background.appendChild(handed);

  const experience = createEl("div", "question");
  experience.appendChild(createEl("span", "label", t("prestudy.experience")));
  const expGrid = createEl("div", "option-grid");
  [
    { value: "Gesture control (phone/watch/AR/VR)", key: "prestudy.experience.gesture" },
    { value: "Drones / RC control", key: "prestudy.experience.drones" },
    { value: "Robotics (classes/projects)", key: "prestudy.experience.robotics" },
    { value: "Wearables (EMG/smartwatch/rings/etc.)", key: "prestudy.experience.wearables" },
    { value: "None of the above", key: "prestudy.experience.none" },
  ].forEach((item) => {
    expGrid.appendChild(optionInput({ type: "checkbox", name: "experience", value: item.value, text: t(item.key) }));
  });
  experience.appendChild(expGrid);
  background.appendChild(experience);

  container.appendChild(background);

  const baseline = createEl("div", "form-section");
  baseline.appendChild(createEl("h3", "", t("prestudy.baseline.title")));
  baseline.appendChild(createEl("p", "muted", t("prestudy.baseline.scale")));
  baseline.appendChild(
    buildLikertGrid("baseline", [
      { key: "comfortable", label: t("prestudy.baseline.comfortable") },
      { key: "worry", label: t("prestudy.baseline.worry") },
      { key: "avoid", label: t("prestudy.baseline.avoid") },
    ], t),
  );
  container.appendChild(baseline);
}

export function renderBlockForm({ elements, t, isPrivateEnvironment }) {
  const container = elements.blockForm;
  if (!container) return;
  container.innerHTML = "";

  if (isPrivateEnvironment()) {
    const interpret = createEl("div", "form-section");
    interpret.appendChild(createEl("h3", "", t("block.interpretation.title")));
    interpret.appendChild(
      fieldInput({
        id: "block-interpretation",
        label: t("block.interpretation.prompt"),
        required: true,
      }),
    );
    const comfort = createEl("div", "question");
    comfort.appendChild(createEl("h4", "label", t("block.comfort.question")));
    comfort.appendChild(createEl("p", "muted", t("block.comfort.scale")));
    comfort.appendChild(buildScaleGrid("block_comfort", 7));
    interpret.appendChild(comfort);
    container.appendChild(interpret);
  } else {
    const comfort = createEl("div", "form-section");
    comfort.appendChild(createEl("h3", "", t("block.comfort.question")));
    comfort.appendChild(createEl("p", "muted", t("block.comfort.scale")));
    comfort.appendChild(buildScaleGrid("block_comfort", 7));
    container.appendChild(comfort);
  }

  const control = createEl("div", "form-section");
  control.appendChild(createEl("h3", "", t("block.control.title")));
  control.appendChild(createEl("p", "muted", t("block.scale.agree")));
  control.appendChild(
    buildLikertGrid("control", [
      { key: "in_control", label: t("block.control.in_control") },
      { key: "expected", label: t("block.control.expected") },
      { key: "confident", label: t("block.control.confident") },
      { key: "recover", label: t("block.control.recover") },
    ], t),
  );
  container.appendChild(control);

  const social = createEl("div", "form-section");
  social.appendChild(createEl("h3", "", t("block.social.title")));
  social.appendChild(createEl("p", "muted", t("block.scale.agree")));
  social.appendChild(
    buildLikertGrid("social", [
      { key: "self_conscious", label: t("block.social.self_conscious") },
      { key: "judged", label: t("block.social.judged") },
      { key: "held_back", label: t("block.social.held_back") },
    ], t),
  );
  container.appendChild(social);

  const justify = createEl("div", "form-section");
  justify.appendChild(createEl("h3", "", t("block.justify.title")));
  justify.appendChild(createEl("p", "muted", t("block.scale.agree")));
  const justifyRows = [
    { key: "social", label: t("block.justify.social") },
  ];
  if (!isPrivateEnvironment()) {
    justifyRows.push(
      { key: "bystanders", label: t("block.justify.bystanders") },
      { key: "predict", label: t("block.justify.predict") },
    );
  }
  justify.appendChild(
    buildLikertGrid("justify", justifyRows, t),
  );
  container.appendChild(justify);

  const workload = createEl("div", "form-section");
  workload.appendChild(createEl("h3", "", t("block.workload.title")));
  workload.appendChild(createEl("p", "muted", t("block.workload.scale")));
  workload.appendChild(
    buildLikertGrid("tlx", [
      { key: "mental", label: t("block.workload.mental") },
      { key: "physical", label: t("block.workload.physical") },
      { key: "effort", label: t("block.workload.effort") },
      { key: "frustration", label: t("block.workload.frustration") },
    ], t),
  );
  const performance = createEl("div", "question");
  performance.appendChild(createEl("span", "label", t("block.performance.label")));
  performance.appendChild(createEl("p", "muted", t("block.performance.scale")));
  performance.appendChild(buildScaleGrid("performance", 7));
  workload.appendChild(performance);
  container.appendChild(workload);

  const diagnostic = createEl("div", "form-section");
  diagnostic.appendChild(createEl("h3", "", t("block.diagnostic.title")));
  const diagField = document.createElement("label");
  diagField.className = "field";
  diagField.appendChild(
    document.createTextNode(t("block.diagnostic.prompt")),
  );
  const diagInput = document.createElement("textarea");
  diagInput.id = "block-diagnostic";
  diagInput.rows = 3;
  diagInput.required = true;
  diagField.appendChild(diagInput);
  diagnostic.appendChild(diagField);
  container.appendChild(diagnostic);
}

export function renderFinalForm({ elements, t }) {
  const container = elements.finalForm;
  if (!container) return;
  container.innerHTML = "";

  const contextNote = createEl("div", "form-section");
  contextNote.appendChild(createEl("h2", "muted", t("final.note.visible_robot")));
  container.appendChild(contextNote);

  const discreetPeople = createEl("div", "form-section");
  discreetPeople.appendChild(createEl("h3", "", t("final.discreet.people.title")));
  const dpGrid = createEl("div", "option-grid");
  [
    { value: "Alone", key: "final.people.alone" },
    { value: "Partner", key: "final.people.partner" },
    { value: "Close friends", key: "final.people.close_friends" },
    { value: "Family", key: "final.people.family" },
    { value: "Colleagues", key: "final.people.colleagues" },
    { value: "Strangers", key: "final.people.strangers" },
  ].forEach((item) => {
    dpGrid.appendChild(optionInput({ type: "checkbox", name: "discreet_people", value: item.value, text: t(item.key) }));
  });
  discreetPeople.appendChild(dpGrid);
  container.appendChild(discreetPeople);

  const discreetLocations = createEl("div", "form-section");
  discreetLocations.appendChild(createEl("h3", "", t("final.discreet.locations.title")));
  const dlGrid = createEl("div", "option-grid");
  [
    { value: "At home", key: "final.location.home" },
    { value: "Lab / office", key: "final.location.lab" },
    { value: "Hallway / campus common area", key: "final.location.hallway" },
    { value: "Sidewalk / street", key: "final.location.street" },
    { value: "Public transport (bus/train)", key: "final.location.transport" },
    { value: "Cafe / restaurant", key: "final.location.cafe" },
    { value: "Other", key: "final.location.other" },
  ].forEach((item) => {
    dlGrid.appendChild(
      optionInput({ type: "checkbox", name: "discreet_locations", value: item.value, text: t(item.key) }),
    );
  });
  const dlOther = document.createElement("input");
  dlOther.type = "text";
  dlOther.id = "discreet-location-other";
  dlOther.placeholder = t("final.location.other_placeholder");
  dlGrid.appendChild(dlOther);
  discreetLocations.appendChild(dlGrid);
  container.appendChild(discreetLocations);

  const expressivePeople = createEl("div", "form-section");
  expressivePeople.appendChild(createEl("h3", "", t("final.expressive.people.title")));
  const epGrid = createEl("div", "option-grid");
  [
    { value: "Alone", key: "final.people.alone" },
    { value: "Partner", key: "final.people.partner" },
    { value: "Close friends", key: "final.people.close_friends" },
    { value: "Family", key: "final.people.family" },
    { value: "Colleagues", key: "final.people.colleagues" },
    { value: "Strangers", key: "final.people.strangers" },
  ].forEach((item) => {
    epGrid.appendChild(optionInput({ type: "checkbox", name: "expressive_people", value: item.value, text: t(item.key) }));
  });
  expressivePeople.appendChild(epGrid);
  container.appendChild(expressivePeople);

  const expressiveLocations = createEl("div", "form-section");
  expressiveLocations.appendChild(createEl("h3", "", t("final.expressive.locations.title")));
  const elGrid = createEl("div", "option-grid");
  [
    { value: "At home", key: "final.location.home" },
    { value: "Lab / office", key: "final.location.lab" },
    { value: "Hallway / campus common area", key: "final.location.hallway" },
    { value: "Sidewalk / street", key: "final.location.street" },
    { value: "Public transport (bus/train)", key: "final.location.transport" },
    { value: "Cafe / restaurant", key: "final.location.cafe" },
    { value: "Other", key: "final.location.other" },
  ].forEach((item) => {
    elGrid.appendChild(
      optionInput({ type: "checkbox", name: "expressive_locations", value: item.value, text: t(item.key) }),
    );
  });
  const elOther = document.createElement("input");
  elOther.type = "text";
  elOther.id = "expressive-location-other";
  elOther.placeholder = t("final.location.other_placeholder");
  elGrid.appendChild(elOther);
  expressiveLocations.appendChild(elGrid);
  container.appendChild(expressiveLocations);

  const compare = createEl("div", "form-section");
  compare.appendChild(createEl("h3", "", t("final.compare.title")));
  compare.appendChild(createEl("p", "muted", t("final.compare.scale")));

  const publicPref = createEl("div", "question");
  publicPref.appendChild(createEl("span", "label", t("final.compare.public")));
  publicPref.appendChild(buildScaleGrid("prefer_public", 7));
  compare.appendChild(publicPref);

  const privatePref = createEl("div", "question");
  privatePref.appendChild(createEl("span", "label", t("final.compare.private")));
  privatePref.appendChild(buildScaleGrid("prefer_private", 7));
  compare.appendChild(privatePref);

  const precise = createEl("div", "question");
  precise.appendChild(createEl("span", "label", t("final.compare.precise")));
  precise.appendChild(buildScaleGrid("more_precise", 7));
  compare.appendChild(precise);

  const embarrass = createEl("div", "question");
  embarrass.appendChild(createEl("span", "label", t("final.compare.embarrassing")));
  embarrass.appendChild(buildScaleGrid("more_embarrassing", 7));
  compare.appendChild(embarrass);

  const justified = createEl("div", "question");
  justified.appendChild(createEl("span", "label", t("final.compare.justified")));
  justified.appendChild(buildScaleGrid("more_justified", 7));
  compare.appendChild(justified);
  container.appendChild(compare);

}

export function collectPrestudyAnswers({ elements, t, setStageError }) {
  const errors = [];
  const ageValue = getInputValue("age");
  const age = ageValue ? Number(ageValue) : null;
  if (!age || age < 13 || age > 99) errors.push(t("error.age"));
  const handedness = getCheckedValue("handedness");
  if (!handedness) errors.push(t("error.handedness"));
  const experience = getCheckedValues("experience");
  if (!experience.length) errors.push(t("error.experience"));

  const baselineComfort = getCheckedValue("baseline_comfortable");
  const baselineWorry = getCheckedValue("baseline_worry");
  const baselineAvoid = getCheckedValue("baseline_avoid");
  if (!baselineComfort || !baselineWorry || !baselineAvoid) errors.push(t("error.baseline"));

  if (errors.length) {
    setStageError(elements.prestudyError, errors[0]);
    return null;
  }

  return {
    age,
    gender: getCheckedValue("gender") || null,
    handedness,
    experience,
    baseline: {
      comfortable: Number(baselineComfort),
      worry: Number(baselineWorry),
      avoid: Number(baselineAvoid),
    },
  };
}

export function collectBlockAnswers({ elements, t, setStageError, currentBlock, sessionEnvironment, isPrivateEnvironment }) {
  const errors = [];
  const interpretation = getInputValue("block-interpretation");
  if (isPrivateEnvironment() && !interpretation) errors.push(t("error.block.interpretation"));
  const comfort = getCheckedValue("block_comfort");
  if (!comfort) errors.push(t("error.block.comfort"));

  const requiredGroups = [
    "control_in_control",
    "control_expected",
    "control_confident",
    "control_recover",
    "social_self_conscious",
    "social_judged",
    "social_held_back",
    "justify_social",
    "tlx_mental",
    "tlx_physical",
    "tlx_effort",
    "tlx_frustration",
    "performance",
  ];
  if (!isPrivateEnvironment()) {
    requiredGroups.push("justify_bystanders", "justify_predict");
  }

  requiredGroups.forEach((name) => {
    if (!getCheckedValue(name)) errors.push(t("error.block.required"));
  });

  const diagnostic = getInputValue("block-diagnostic");
  if (!diagnostic) errors.push(t("error.block.diagnostic"));

  if (errors.length) {
    setStageError(elements.blockError, errors[0]);
    return null;
  }

  const payload = {
    block: {
      preset: currentBlock,
      environment: sessionEnvironment,
    },
    comfort: Number(comfort),
    control_confidence: {
      in_control: Number(getCheckedValue("control_in_control")),
      expected: Number(getCheckedValue("control_expected")),
      confident: Number(getCheckedValue("control_confident")),
      recover: Number(getCheckedValue("control_recover")),
    },
    social: {
      self_conscious: Number(getCheckedValue("social_self_conscious")),
      judged: Number(getCheckedValue("social_judged")),
      held_back: Number(getCheckedValue("social_held_back")),
    },
    justification: {
      social: Number(getCheckedValue("justify_social")),
      bystanders: isPrivateEnvironment() ? null : Number(getCheckedValue("justify_bystanders")),
      predict: isPrivateEnvironment() ? null : Number(getCheckedValue("justify_predict")),
    },
    workload: {
      mental: Number(getCheckedValue("tlx_mental")),
      physical: Number(getCheckedValue("tlx_physical")),
      effort: Number(getCheckedValue("tlx_effort")),
      frustration: Number(getCheckedValue("tlx_frustration")),
    },
    performance: Number(getCheckedValue("performance")),
    diagnostic,
  };
  if (isPrivateEnvironment()) {
    payload.interpretation = interpretation;
  }
  return payload;
}

export function collectFinalAnswers({ elements, t, setStageError }) {
  const errors = [];
  const discreetPeople = getCheckedValues("discreet_people");
  const discreetLocations = getCheckedValues("discreet_locations");
  const expressivePeople = getCheckedValues("expressive_people");
  const expressiveLocations = getCheckedValues("expressive_locations");

  if (!discreetPeople.length) errors.push(t("error.final.people.discreet"));
  if (!discreetLocations.length) errors.push(t("error.final.locations.discreet"));
  if (!expressivePeople.length) errors.push(t("error.final.people.expressive"));
  if (!expressiveLocations.length) errors.push(t("error.final.locations.expressive"));

  const discreetOther = getInputValue("discreet-location-other");
  if (discreetOther) discreetLocations.push(`Other: ${discreetOther}`);
  const expressiveOther = getInputValue("expressive-location-other");
  if (expressiveOther) expressiveLocations.push(`Other: ${expressiveOther}`);

  const requiredRadios = [
    "prefer_public",
    "prefer_private",
    "more_precise",
    "more_embarrassing",
    "more_justified",
  ];
  requiredRadios.forEach((name) => {
    if (!getCheckedValue(name)) errors.push(t("error.final.compare"));
  });

  if (errors.length) {
    setStageError(elements.finalError, errors[0]);
    return null;
  }

  function readScale(name) {
    const value = getCheckedValue(name);
    return value ? Number(value) : null;
  }

  return {
    discreet_people: discreetPeople,
    discreet_locations: discreetLocations,
    expressive_people: expressivePeople,
    expressive_locations: expressiveLocations,
    comparison: {
      prefer_public: readScale("prefer_public"),
      prefer_private: readScale("prefer_private"),
      more_precise: readScale("more_precise"),
      more_embarrassing: readScale("more_embarrassing"),
      more_justified: readScale("more_justified"),
    },
  };
}
