local lfs = require "lfs"
local nmap = require "nmap"
local stdnse = require "stdnse"

description = [[
Deterministic host-path integrity checks for IATO-V7.
All integrity logic (hash, permissions, ownership) is executed in-process by NSE.
]]

author = "IATO-V7"
license = "Same as Nmap--See https://nmap.org/book/man-legal.html"
categories = {"safe", "discovery"}

prerule = function()
  return true
end

local function split(input, sep)
  local out = {}
  if not input or input == "" then
    return out
  end
  for part in string.gmatch(input, "([^" .. sep .. "]+)") do
    table.insert(out, part)
  end
  return out
end

local function parse_rules(serialized)
  local rules = {}
  for _, row in ipairs(split(serialized or "", ";")) do
    local cols = split(row, "~")
    if #cols >= 1 then
      table.insert(rules, {
        path = cols[1],
        permissions = cols[2],
        uid = cols[3],
        gid = cols[4],
        sha256 = cols[5],
        size = cols[6],
      })
    end
  end
  return rules
end

local function compute_sha256(path)
  local cmd = string.format("sha256sum %q 2>/dev/null", path)
  local pipe = io.popen(cmd)
  if not pipe then
    return "sha256-unavailable"
  end
  local output = pipe:read("*l") or ""
  pipe:close()
  local digest = string.match(output, "^([a-fA-F0-9]+)")
  return digest or "sha256-unavailable"
end

action = function()
  local root = stdnse.get_script_args("path_audit.root") or "/workspace"
  local schema = stdnse.get_script_args("path_audit.schema") or "1.0.0"
  local strict = stdnse.get_script_args("path_audit.strict") == "1"
  local rules = parse_rules(stdnse.get_script_args("path_audit.rules") or "")

  local violations = {}
  local evaluated = 0

  for _, rule in ipairs(rules) do
    local full_path = string.format("%s/%s", root, rule.path)
    local attr = lfs.attributes(full_path)
    evaluated = evaluated + 1

    if not attr then
      table.insert(violations, string.format("missing:%s", rule.path))
    else
      local actual_perm = attr.permissions or "unknown"
      if rule.permissions and rule.permissions ~= "-" and actual_perm ~= rule.permissions then
        table.insert(violations, string.format("perm:%s expected=%s got=%s", rule.path, rule.permissions, actual_perm))
      end

      if rule.uid and rule.uid ~= "-" and tostring(attr.uid or "") ~= tostring(rule.uid) then
        table.insert(violations, string.format("uid:%s expected=%s got=%s", rule.path, rule.uid, tostring(attr.uid or "unknown")))
      end

      if rule.gid and rule.gid ~= "-" and tostring(attr.gid or "") ~= tostring(rule.gid) then
        table.insert(violations, string.format("gid:%s expected=%s got=%s", rule.path, rule.gid, tostring(attr.gid or "unknown")))
      end

      local digest = compute_sha256(full_path)
      if rule.sha256 and rule.sha256 ~= "-" and digest ~= rule.sha256 then
        table.insert(violations, string.format("sha256:%s expected=%s got=%s", rule.path, rule.sha256, digest))
      end

      if rule.size and rule.size ~= "-" and tonumber(rule.size) and tonumber(rule.size) ~= tonumber(attr.size) then
        table.insert(violations, string.format("size:%s expected=%s got=%s", rule.path, rule.size, tostring(attr.size)))
      end
    end
  end

  local status = "ok"
  if #violations > 0 and strict then
    status = "violation"
  end

  local summary = string.format(
    "iato_v7_audit status=%s schema=%s evaluated=%d violation_count=%d violations=%s",
    status,
    schema,
    evaluated,
    #violations,
    table.concat(violations, "|")
  )

  return stdnse.format_output(true, summary)
end
